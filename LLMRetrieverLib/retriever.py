from langchain_openai.chat_models.azure import AzureChatOpenAI
from concurrent.futures import ThreadPoolExecutor

class llm_retriever:
    @staticmethod
    # Function to analyze each chunk with the LLM and query relevance
    def analyze_chunk_with_llm(llm: AzureChatOpenAI, chunk: str, query: str):    
        """
        Function to analyze each chunk with the LLM and query relevance.
         Args:
            llm (AzureChatOpenAI): Azure Chat OpenAI Model.
            chunk: the chunk to analyze to get the relevance with user question.
            query: the question to compare with the chunk. 
        """
        # Define OpenAI prompt    
        messages = [
            {"role": "system", "content": "You are an assistant helping determine relevance of text to a user question."},
            {"role": "user", "content": f"Given the user question '{query}', is the following text relevant and can be usefull to answer to the question?\n\n{chunk}\n\nAnswer 'yes' or 'no'."}
        ]  
        response = llm.invoke(messages)
        # Return boolean and chunk
        return response.content.strip().lower() == 'yes', chunk  

    @staticmethod
    # Multi-threading function to distribute chunk analysis across thread
    def process_chunks_in_parallel(llm: AzureChatOpenAI, question: str, database, threads_num: int = 4):
        """
        Multi-threading function to distribute chunk analysis across multiple thread
        
        The main idea is to assign to each thread a subset of chunks extracted from the database 
        and have each thread carry out an analysis on the relevance of the chunk with based on user question.
        
        Args:
            llm (AzureChatOpenAI): Azure Chat OpenAI Model.
            question (str): user query
            database (array): an array with a list of chunk in string format. 
            threads_num (int, optional): number of thread we want to use in analysis. Defaults to 4.
        """
        total_chunks = len(database)
        # calculate number of chunks for each thread
        chunk_per_thread = total_chunks // threads_num
        relevant_chunks = []
        
        # Function for each thread to process its assigned chunks
        def thread_task(thread_id: int):
            start = thread_id * chunk_per_thread
            end = start + chunk_per_thread if thread_id != threads_num - 1 else total_chunks
            thread_relevant_chunks = []
            
            for chunk in database[start:end]:
                is_relevant, useful_chunk = llm_retriever.analyze_chunk_with_llm(llm, chunk, question)
                if is_relevant:
                    thread_relevant_chunks.append(useful_chunk)
                    
            return thread_relevant_chunks
        
        # Use ThreadPoolExecutor to handle threads concurrently
        with ThreadPoolExecutor(max_workers=threads_num) as executor:
            futures = [executor.submit(thread_task, thread_id) for thread_id in range(threads_num)]
            for future in futures:
                relevant_chunks.extend(future.result())  # Gather relevant chunks from all threads
        
        return relevant_chunks

    @staticmethod
    # Function to generate the final response using LLM with the relevant chunks
    def generate_final_answer_with_llm(llm: AzureChatOpenAI, relevant_chunks: list[str], question: str):
        """
            This function take relevant chunks and try to give an answer
        Args:
            llm (AzureChatOpenAI): Azure Chat OpenAI Model.
            relevant_chunks (list[str]): useful chunks to answer the question
            question (str): user question
        """
        #combine relevant chunks in a string
        combined_chunks = "\n".join(relevant_chunks)
        #openAI prompt 
        messages = [
            {"role": "system", "content": "You are an assistant providing comprehensive answers based on relevant information."},
            {"role": "user", "content": f"Using the following relevant chunks, answer the user query '{question}':\n\n{combined_chunks}\n\nProvide a comprehensive answer."}
        ]
        final_response = llm.invoke(messages)
        # return only the content of LLM response
        return final_response.content
