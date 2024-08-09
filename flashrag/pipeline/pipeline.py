from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template
        
        # load NLI classifier
        from transformers import pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=1)

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass
    
    def compute_unfairness_distribution(self, dataset, retrieval_results, type=None):
        ### Compute the distribution of protected, unprotected, and neutral groups after retrieval ########
        datasets = dataset.data
        all_data_dist_scores = []  # A list of all dist_score for each query 
        
        for idx, data in enumerate(tqdm(datasets, desc="Computing unfairness distribution")):
            candidate_labels = data.metadata["candidate_labels"]       
            if type == "sc":
                ### handle selective-context
                retrieved_docs = [retrieval_results[idx]]
            else:
                retrieved_docs = retrieval_results[idx]  # All N docs retrieved for a query
            
            switch_list = False
            if data.metadata["target_loc"] == 2.0:
                switch_list = True
            
            def process_doc(doc):
                if type == "sc":
                    # handle selective-context 
                    classifier_res = self.classifier(doc, candidate_labels, multi_label=True)
                else:    
                    classifier_res = self.classifier(doc['contents'], candidate_labels, multi_label=True)
                label_score_dict = dict(zip(classifier_res['labels'], classifier_res['scores']))
                raw_doc_dist_score = [label_score_dict[label] for label in candidate_labels]
                doc_dist_score_sum = sum(raw_doc_dist_score)
                doc_dist_score = [s / doc_dist_score_sum for s in raw_doc_dist_score]  # Normalize scores
                return doc_dist_score
            
            with ThreadPoolExecutor() as executor:
                query_dist_scores = list(executor.map(process_doc, retrieved_docs))
            
            query_dist_score = [sum(x[i] for x in query_dist_scores) / len(query_dist_scores) for i in range(len(query_dist_scores[0]))]
            
            ### Switch number so that protected group is always at loc 0 ##
            if switch_list:
                query_dist_score[0], query_dist_score[2] = query_dist_score[2], query_dist_score[0]
                switch_list = False
            
            all_data_dist_scores.append(query_dist_score)
        
        all_data_dist_score = [sum(x[i] for x in all_data_dist_scores) / len(all_data_dist_scores) for i in range(len(all_data_dist_scores[0]))]
        print("unfairness distribution: ", all_data_dist_score)

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            raw_pred = dataset.pred
            processed_pred = [pred_process_fun(pred) for pred in raw_pred]
            dataset.update_output("raw_pred", raw_pred)
            dataset.update_output("pred", processed_pred)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        self.generator = None
        if config["refiner_name"] is not None:
            # For refiners other than kg, do not load the generator for now to save memory
            if "kg" in config["refiner_name"].lower():
                self.generator = get_generator(config) if generator is None else generator
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None
            self.generator = get_generator(config) if generator is None else generator

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question

        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

        ## compute unfairness distribution
        self.compute_unfairness_distribution(dataset, retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                if "selective-context" in self.refiner.name:
                    ## compute sc unfairness distribution
                    self.compute_unfairness_distribution(dataset, refine_results, type="sc")
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
            ]
        dataset.update_output("prompt", input_prompts)
        # print("prompt--------", input_prompts)

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc for doc in docs])
        # delete used refiner to release memory
        if self.refiner:
            if "kg" in self.config["refiner_name"].lower():
                self.generator = self.refiner.generator
            else:
                self.generator = get_generator(self.config)
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)
        self.judger = get_judger(config)
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)

        self.sequential_pipeline = SequentialPipeline(
            config, prompt_template, retriever=self.retriever, generator=self.generator
        )

        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        #### Here is a bug, dataset_split[True], there may not be any True
        pos_dataset, neg_dataset = dataset_split[True], dataset_split[False]

        ######### the unfairness distribution idea for skr is that those pos_dataset will 
        ### NOT use external knowledge, if this part has more unfairness, then compared to
        ### just use rag, it means unfairness increases. 
        print("skr: ")
        print("pos len: ", len(pos_dataset.data))
        print("neg len: ", len(neg_dataset.data))
        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class AdaptivePipeline(BasicPipeline):
    def __init__(
        self,
        config,
        norag_template=None,
        single_hop_prompt_template=None,
        multi_hop_prompt_template=None,
    ):
        super().__init__(config)
        # load adaptive classifier as judger
        self.judger = get_judger(config)

        retriever = get_retriever(config)
        generator = get_generator(config)

        # Load three pipeline for three types of query: naive/single-hop/multi-hop
        from flashrag.pipeline import IRCOTPipeline

        if norag_template is None:
            norag_templete = PromptTemplate(
                config=config,
                system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
                user_prompt="Question: {question}",
            )
        self.norag_pipeline = SequentialPipeline(
            config,
            prompt_template=norag_templete,
            retriever=retriever,
            generator=generator,
        )

        self.single_hop_pipeline = SequentialPipeline(
            config,
            prompt_template=single_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

        self.multi_hop_pipeline = IRCOTPipeline(
            config,
            prompt_template=multi_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: choice result representing which pipeline to use(e.g. A, B, C)
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        for symbol, symbol_dataset in dataset_split.items():
            if symbol == "A":
                symbol_dataset = self.norag_pipeline.naive_run(symbol_dataset, do_eval=False)
            elif symbol == "B":
                symbol_dataset = self.single_hop_pipeline.run(symbol_dataset, do_eval=False)
            elif symbol == "C":
                symbol_dataset = self.multi_hop_pipeline.run(symbol_dataset, do_eval=False)
            else:
                assert False, "Unknown symbol!"

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset
