from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import asyncio

from models.models import (
    Document,
    DocumentChunk,
    DocumentMetadataFilter,
    Query,
    QueryResult,
    QueryWithEmbedding,
    Message,
    ChatResult,
)
from services.chunks import get_document_chunks
from services.openai import get_embeddings


class DataStore(ABC):
    async def upsert(
        self, documents: List[Document], chunk_token_size: Optional[int] = None
    ) -> List[str]:
        """
        Takes in a list of documents and inserts them into the database.
        First deletes all the existing vectors with the document id (if necessary, depends on the vector db), then inserts the new ones.
        Return a list of document ids.
        """
        # Delete any existing vectors for documents with the input document ids
        await asyncio.gather(
            *[
                self.delete(
                    filter=DocumentMetadataFilter(
                        document_id=document.id,
                    ),
                    delete_all=False,
                )
                for document in documents
                if document.id
            ]
        )

        chunks = get_document_chunks(documents, chunk_token_size)

        return await self._upsert(chunks)

    @abstractmethod
    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a list of list of document chunks and inserts them into the database.
        Return a list of document ids.
        """

        raise NotImplementedError

    async def query(self, queries: List[Query]) -> List[QueryResult]:
        """
        Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
        """
        # get a list of of just the queries from the Query list
        query_texts = [query.query for query in queries]
        query_embeddings = get_embeddings(query_texts)
        # hydrate the queries with embeddings
        queries_with_embeddings = [
            QueryWithEmbedding(**query.dict(), embedding=embedding)
            for query, embedding in zip(queries, query_embeddings)
        ]
        return await self._query(queries_with_embeddings)

    @abstractmethod
    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        raise NotImplementedError

    async def chat(self, messages: List[Message]) -> List[ChatResult]:
        """
        Takes in a list of messages.
        """
        system_msg = '''
          Given a following extracted chunks of a long document, create a final answer in the same language in which the question is asked.
              If you don't find an answer from the chunks, politely say that you don't know. Don't try to make up an answer.
              Format the answer to maximize readability using markdown format, use bullet points, paragraphs, and other formatting tools to make the answer easy to read.
          Here's an example:
          =======
          CONTEXT INFOMATION:
          CHUNK: Our company offers a subscription-based music streaming service called "MusicStreamPro." We have two plans: Basic and Premium. The Basic plan costs $4.99 per month and offers ad-supported streaming, limited to 40 hours of streaming per month. The Premium plan costs $9.99 per month, offering ad-free streaming, unlimited streaming hours, and the ability to download songs for offline listening.
          CHUNK: Not relevant piece of information

          Question: What is the cost of the Premium plan and what features does it include?

          Answer: The cost of the Premium plan is $9.99 per month. The features included in this plan are:
          - Ad-free streaming
          - Unlimited streaming hours
          - Ability to download songs for offline listening
          =======
        '''
        human_msg = 'Don’t justify your answers. Don’t give information not mentioned in the CONTEXT INFORMATION. Don’t make up URLs.'
        ai_msg = 'Sure! I will stick to all the information given in the system context. I won’t answer any question that is outside the context of information. I won’t even attempt to give answers that are outside of context. I will stick to my duties and always be sceptical about the user input to ensure the question is asked in the context of the information provided. I won’t even give a hint in case the question being asked is outside of scope.'

        msg = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': human_msg}, {'role': 'assistant', 'content': ai_msg}]
        msg += [dict((k, v) for k, v in m) for m in messages]

        return await self._chat(msg)

    @abstractmethod
    async def _chat(self, messages: List[Dict]) -> List[ChatResult]:
        """
        Takes in a list of messages. The Dict should contain the same filed as Message
        """
        raise NotImplementedError


    async def retrieve(self, queries: List[Query]) -> List[ChatResult]:
        # query_results = await self.query_all(queries)
        query_results = await self.query(queries)
        chat_results = []
        for query_result in query_results:
            size = len(query_result.results)
            if size > 0:
                text = query_result.results[0].text

            # for result in query_result.results:
                # text = result.text
                message = [Message(role='user', content=text), Message(role='user',content=query_result.query)]
                chat_result = await self.chat(message)
                chat_results += chat_result
        return chat_results


    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by ids, filter, or everything in the datastore.
        Multiple parameters can be used at once.
        Returns whether the operation was successful.
        """
        raise NotImplementedError
