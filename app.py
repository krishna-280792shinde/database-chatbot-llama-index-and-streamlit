import os
import streamlit as st
from sqlalchemy import create_engine
from streamlit_extras.add_vertical_space import add_vertical_space
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.callbacks import CallbackManager, TokenCountingHandler
import tiktoken

from llama_index import SQLDatabase, ServiceContext
from llama_index.llms import OpenAI

os.environ["OPENAI_API_KEY"]="YOUR_OPENAI_API_KEY"

with st.sidebar:
    st.title("Database Interactive AI Chatbot")
    st.markdown('''
    ## About
    Elevate database interactions with our AI Chatbot & LLM
    - 💬 Chat with your databse
 
    ''')
    add_vertical_space(15)
    st.write('Made with ❤️ by  [Krishna Shinde](https://www.linkedin.com/in/krishna-shinde-3064a1258/)')
    
def create_sql_database(db_uri):
    if db_uri:
        db_engine = create_engine(db_uri)
        sql_db = SQLDatabase(db_engine)
        return sql_db
    else:
        return None

def run_query(sql_database, query, llm):
    if sql_database:
        token_counter = TokenCountingHandler(tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode)
        callback_manager = CallbackManager([token_counter])
        service_context= ServiceContext.from_defaults(llm=llm, callback_manager=callback_manager)
        query_engine = NLSQLTableQueryEngine(sql_database=sql_database, service_context=service_context)
        response = query_engine.query(query)
        return response.response
    else:
        return "No valid SQL database provided."

def main():
    llm=OpenAI(temperature=0.7, model="gpt-3.5-turbo")
    st.title("Chat with your Database")
    db_uri = st.text_input("Provide your database connection string")
    if db_uri:
        if "messages" not in st.session_state.keys(): 
            st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about your database!"}]

        if prompt := st.chat_input("Your question"): 
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages: 
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    sql_db = create_sql_database(db_uri)
                    response = run_query(sql_db, prompt, llm)
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)
    else:
        st.stop()

if __name__ == "__main__":
    main()
