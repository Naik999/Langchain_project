from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from flask import Flask,render_template,redirect,request, session, url_for
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader,YoutubeLoader,TextLoader
from docx import Document
from langchain.llms import OpenAI
import os
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import requests
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase


app=Flask(__name__)
@app.route('/')
def home():
    return render_template("mainfile.html")
@app.route('/documentanalyse')
def document_analyse():
    return render_template("in.html")

@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        if 'fname' not in request.files:
            return 'No file part'
        file = request.files['fname']
        file_name='C:/Users/19m61/OneDrive/Desktop/langchain/static/' + file.filename
        file.save(file_name)
        query = request.form['query']
        if os.path.exists(file_name):
            if file.filename.endswith(".txt"):
                loader = TextLoader(file_path=file_name)
                pages = loader.load()
            elif file.filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path=file_name)
                pages = loader.load_and_split()
            elif file.filename.endswith(".csv"):
                loader = CSVLoader(file_path=file_name)
                pages = loader.load()
            else:
                output= "your file can't be processed"
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(pages)
        vector = FAISS.from_documents(documents, embeddings)
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>

        Question: {input}""")
        llm=ChatOpenAI(temperature=0.9)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input":f"{query}" })
        # session['file_name']=file.filename
        
    return render_template("index.html",outputt=response['answer'])
@app.route("/qtq",methods=['GET','POST'])
def QTQ():
    if request.method=='POST':
        ques = request.form['qn']
        db = SQLDatabase.from_uri('mysql://root:Ppadma%401234@localhost/api_base')
        llm = ChatOpenAI( temperature=0)
        chain = create_sql_query_chain(llm,db)
        res = chain.invoke({"question": ques})
        return render_template('qr.html',response=res,q=ques,above='Your output is Above')
    return render_template('QTQ.html')


@app.route('/Generate_story' , methods= ['POST' , 'GET'])
def Generate_story():

    if request.method=='POST':
        prompt = request.form['prompt']
        llm = OpenAI(temperature=0.7)

        story_template = PromptTemplate(
        input_variables=['story'],
        template='Write a story according to the cast which are leveraging the wikipedia research as well :{story}'
        )

        story_chain = LLMChain(llm=llm, prompt=story_template, verbose=True)
        outputs = story_chain.run(story=prompt)
        print(outputs)

        with open ('story.doc' , "w") as data:
            data.write(outputs)

        

        prompt_img = PromptTemplate(
            input_variables=["image_desc"],
            template="generate a image description: {image_desc}",
        )

        prompt_i = prompt
        chain = LLMChain(llm=llm, prompt=prompt_img )

        image_url = DallEAPIWrapper(model="dall-e-2").run(chain.run(image_desc=prompt_i))
        response = requests.get(image_url , stream=True)
        with open("image.png", "wb") as img:
            for new in response:  
                img.write(new)

        return render_template('rindex.html' , story=outputs ,image_url= image_url )
    
    return render_template('rindex.html' , story=None)
@app.route('/videosummarize')
def videosummarize():
    return render_template("youtube.html")
@app.route('/summary', methods=['POST'])
def youtube_bot():
    youtube_url = request.form.get("youtube_url")
    user_input = request.form.get("query")

    # Optimized loading and processing:
    llm = ChatOpenAI(openai_api_key="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")
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    doc = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    document = splitter.split_documents(doc)
    embedding = OpenAIEmbeddings(openai_api_key="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")
    vectordb = FAISS.from_documents(document, embedding)
    retriever = vectordb.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "answer the following questions based on the provided context:{context}"),
        ("user", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    response = retrieval_chain.invoke({"input": user_input})

    # Re-render the original form with the response
    return render_template('youtube.html', response=response['answer'])

@app.route('/petgenerator' , methods=['POST' , 'GET'])
def generate_pet_name():
    if request.method=='POST':
        pet_name = request.form['name']
        pet_color = request.form['color']
     
        llm = OpenAI(temperature=0.9)  

        promp_template_name = PromptTemplate(
            input_variables= ['animal_type','pet_color'],
            template="i have a {animal_type} pet and i want to a cool name for it is {pet_color} in color .Suggest me five cool names for my pet."
        )

        name_chain =LLMChain(llm=llm,prompt=promp_template_name, output_key= "pet_name" )


        response = name_chain({'animal_type': pet_name,'pet_color': pet_color})
        return render_template('petresult.html' , pet_name = response['pet_name'] )
    
    return render_template ('pet.html')


if __name__=="__main__":
    app.run(port="8000",host='0.0.0.0',  debug=True)
                