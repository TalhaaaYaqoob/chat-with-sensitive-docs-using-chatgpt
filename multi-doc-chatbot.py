import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain.chains import create_structured_output_runnable
import pprint
from presidio_analyzer import Pattern, PatternRecognizer
from faker import Faker
from presidio_anonymizer.entities import OperatorConfig
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


anonymizer = PresidioReversibleAnonymizer(analyzed_fields=["EMAIL_ADDRESS"],add_default_faker_operators=True)

fake = Faker()
load_dotenv('.env')

# Define a new Pydantic model with field descriptions and tailored for Twitter.
class patient_details(BaseModel):
    Query: str = Field(description="query in the account info in the document")
    chart_no: str=Field(discription="chart no in the document")
    DOB: str = Field(discription="date of birth of the patient")
    name: str = Field(description="Full name of the patient")
    MRN : str =Field(description="MRN of the patient")
    age: str = Field(description="Age of the patient.")
    account_no: str = Field(description="account no of the patient")
    Vital_Signs: List[str]=Field(description="list the Vital Signs of the patient with values")

parser = PydanticOutputParser(pydantic_object=patient_details)

prompt1 = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(
            "answer the users question as best as possible.\n{format_instructions}\n{question}"
        )
    ],
    input_variables=["question"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)


@st.cache
def fake_account_no(_=None):
    return fake.bothify(text="??##########").upper()

def main():
    st.title('DocBot - Your Document Assistant')
    st.write('Welcome to DocBot! Upload your PDF document below and start interacting.')

    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

    if uploaded_file is not None:
        temp_folder = "temp"
        os.makedirs(temp_folder, exist_ok=True)

        with open(os.path.join(temp_folder, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("PDF uploaded successfully!")

        st.subheader("Chat Interface")
        st.write("Ask DocBot anything:")
        query = st.text_input("Question:")
        
        if st.button("Ask"):
            if query:
                answer = get_answer(query, os.path.join(temp_folder, uploaded_file.name))
                st.write("DocBot:", answer)
            else:
                st.warning("Please enter a question.")

        shutil.rmtree(temp_folder)

def get_answer(query, pdf_path):
    loader = PyPDFLoader(pdf_path)
    text = loader.load()


    # Define the regex patterns

    account_number_pattern = Pattern(
        name="account_number_pattern",
        regex="AB\d{10}",
        score=1,
    )

    account_number_recognizer = PatternRecognizer(
        supported_entity="ACCOUNT_NUMBER", patterns=[account_number_pattern]
    )

    anonymizer.add_recognizer(account_number_recognizer)
    anonymizer.reset_deanonymizer_mapping()

    new_operators = {
    "ACCOUNT_NUMBER": OperatorConfig("custom", {"lambda": fake_account_no}),
                        }

    anonymizer.add_operators(new_operators)

    anonymizer.reset_deanonymizer_mapping()

    documents = []
    for doc in text:
        doc.page_content = anonymizer.anonymize(doc.page_content)
        documents.append(doc.page_content)

    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=10)
    documents = text_splitter.split_text(" ".join(documents))

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(documents, embeddings)
    retriever = docsearch.as_retriever()



    template = """Answer the question based only on the following context:
    {context}

    Question: {anonymized_question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')


    document_query = "extract the details from this document: " + str(documents)

    _input = prompt1.format_prompt(question=document_query)

    output = llm(_input.to_messages())
    parsed = parser.parse(output.content)

    # pprint.pprint(anonymizer.deanonymizer_mapping)
    print(output.content)
    print(parsed)


    # pprint.pprint(anonymizer.deanonymizer_mapping)


    _inputs = RunnableParallel(
        question=RunnablePassthrough(),
        anonymized_question=RunnableLambda(anonymizer.anonymize),
    )

    anonymizer_chain = (
        _inputs
        | {
            "context": itemgetter("anonymized_question") | retriever,
            "anonymized_question": itemgetter("anonymized_question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    

    chain_with_deanonymization = anonymizer_chain | RunnableLambda(anonymizer.deanonymize)
    result=chain_with_deanonymization.invoke(output.content)
    print('..........',result)

    return parsed


if __name__ == "__main__":
    main()
