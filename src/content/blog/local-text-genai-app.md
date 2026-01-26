---
title: "Building a Local GenAI Application using TinyLlama"
description: "Lighweight AI applications that can run locally"
date: 2025-11-28
tags: ["AI", "TinyLlama", "FastAPI", "Streamlit"]
readTime: 15
---

The other day I thought of building a very simple and minimal GenAI chat application, using Python only. Well naturally, **FastAPI** was my premier choice for 
creating the backend and serving the model. As for keeping it minimal, what better to choose than **Streamlit**. 

Now that the choice for serving the model and client was decided, the only thing that remained was, 

> **"Which model do I use? Do I go for some API?"**

But that came across a very, very common paradigm of building such chatbots. So lets tweak it a bit!

I decided to use a local, and a very lightweight model, for the generative task. I mean, I got a 1650 with a 10th gen I5, no way im running Deepseek on this one!

These struggles (pain!) made **TinyLlama's 1.1B** chat model, the perfect choice for this application. 

With everything in place,
* **TinyLlama** for text generation,
* **FastAPI** for serving the model,
* **Streamlit** for client.

we are ready to begin coding this application. 
I will be using **uv** as my package manager for this application. If you do not have **uv** installed, you can hop onto this website

[https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

and install it as per your operating system. 

Once that's done, you can verify the installation by running
```bash
uv --version
```
If you get an output similar to this 
```bash
uv 0.9.8
```
congratulations, your uv installation was successful. 

Lets create the directory where we will place all of our code and init a uv repository
```bash
mkdir tinyllama-chat-app
cd tinyllama-chat-app
uv init
```

Once this is done, you will see quite a bit files created by uv. Nothing to be overwhelmed by, these are just files uv uses to setup your environment. For example 
**pyproject.toml** is similar to **requirements.txt**, it contains all the libraries along with the version needed to setup this project, along with other metadata.

Now lets create the environment we will work in by running uv sync
```bash
uv sync
```
```bash
# To activate venv

# 1. Windows CMD 
venv/Scripts/activate

# 2. Windows Powershell
venv\Scripts\Activate.ps1

# 3. Linus/WSL/macOS
source venv/bin/activate
```
This will create a python environment by the name **venv** and activate it. This is where all the libraries you install for this project will live. Why is this preferred? Mainly because this isolates your project from global dependencies, which may cause verison conflicts etc etc. In a nutshell, 

> **env -> good** 

> **global libs -> bad**

Now its time to install all the necessary libraries for this application
```bash
uv add transformers torch streamlit fastapi uvicorn[standard] requests
```

Now all that's left to setup is the project structure, so lets get right into it
```
tinyllama-chat-app/
├── venv/
├── .gitignore
├── .python-version
├── main.py              # this is where the FastAPI service will live
├── model.py             # this is where the TinyLlama model will live
├── client.py            # this is where the Streamlit application will live
├── pyproject.toml
├── README.md
├── uv.lock

```

Now comes the most interesting part of this project, the code (laughs in evil!).

We'll start off by adding **TinyLlama**, or more precisely **TinyLlama/TinyLlama-1.1B-Chat-v1.0**. You can access the model card from HuggingFace from this link, 

[https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

We will integrate it through HF's transformers library that we imported while setting up the project.

```python
# model.py

import torch
from transformers import Pipeline, pipeline

# Checks system for CUDA, if available, uses GPU else defaults to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We load the model using pipeline, and set the model name, as given in HF.
def load_tinyllama():
    pipe = pipeline(
        "text-generation",  # we set the task as "text-generation"
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,  # Here we set the type, for precision
        device=device  
    )
    return pipe
```

We have successfully loaded the model, now its time to make use of it to generate text. 

```python
# model.py

def generate_text(
        pipe: Pipeline,
        prompt: str,
        temperature: float = 0.7
) -> str :
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        },
    ]

    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    predictions = pipe(
        prompt,
        temperature=temperature,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    output = predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1]
    return output
```

This completes our TinyLlama setup and text generation utility function. If you are overwhelmed by the parameters we used to get the prediction, don't worry, I was too. But these are just the params we adjust to get the quality, length, type of response we want from the model. More on these in a later blog!

Now it's time to setup the FastAPI server, so let's dive right into it

```python
# main.py

from fastapi import FastAPI
from model import load_tinyllama, generate_text

app = FastAPI()

@app.get("/generate/text")
def serve_tinyllama_controller(prompt: str) -> str:
    pipe = load_tinyllama()
    output = generate_text(pipe, prompt)
    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

Aaand it's as simple as that, this creates a very simple server that serves our TinyLlama model over the endpoint
```
http://localhost:8000/generate/text
```

This leaves just one task, the Streamlit client. As you will see in the next code block, setting up a minimalistic UI is as easy as setting up the model and server.

```python
# client.py

import requests
import streamlit as st  

st.title("TinyLlama Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What can I help you with?"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.text(prompt)
        
    response = requests.get(
        f"http://localhost:8000/generate/text",
        params={"prompt": prompt}
    )
    response.raise_for_status()

    with st.chat_message("assistant"):
        st.markdown(response.text)
```

And this wraps up our entire application. Now all that's left is to run it and test it out. To do that, you'll need to open two terminal windows, or split terminal if you are using VSCode. 
```bash
# Firing up the FastAPI server
python main.py

# this will serve our model over localhost:8000
```

```bash
# Starting up the streamlit client
streamlit client.py

# this will automatically open up the client in a new tab in your browser
```

And there you have it, you just successfully built an entirely local ChatBot using TinyLlama, FastAPI and Streamlit. But you are not limited to just text generation, there are plenty of tiny models for audio, image, video generation for you to use, so feel free to check them out on HuggingFace. 

