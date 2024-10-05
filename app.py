from flask import Flask,render_template,request
import google.generativeai as genai
import os
import numpy as np
import textblob


model = genai.GenerativeModel("gemini-1.5-flash")
api = os.getenv("MAKERSUITE")
api_key = "AIzaSyD1OgO6S8q23rtpRqwHbyICzCyi5OebJ-E"
genai.configure(api_key=api_key)

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/prediction_DBS",methods=["GET","POST"])
def prediction_DBS():
    return(render_template("prediction_DBS.html"))

@app.route("/prediction_result_DBS",methods=["GET","POST"])
def prediction_result_DBS():
    q = float(request.form.get("q"))
    r = (-50.6 * q) + 90.2
    return(render_template("prediction_result_DBS.html",r=r))

@app.route("/predict_credit",methods=["GET","POST"])
def predict_credit():
    return(render_template("predict_credit.html"))

@app.route("/predict_result_credit",methods=["GET","POST"])
def predict_result_credit():
    q = float(request.form.get("q"))
    r = (-9.28978636e-05* q) + 1.14049728
    r = np.where (r>0.5, "Creditable","Not Creditable")
    return(render_template("predict_result_credit.html",r=r))

@app.route("/sentiment_analysis",methods=["GET","POST"])
def sentiment_analysis():
    return(render_template("sentiment_analysis.html"))

@app.route("/sentiment_analysis_result",methods=["GET","POST"])
def sentiment_analysis_result():
    q = request.form.get("q")
    r = textblob.Textblob(q).sentiment
    return(render_template("sentiment_analysis_result",r=r))

@app.route("/faq",methods=["GET","POST"])
def faq():
    return(render_template("faq.html"))

@app.route("/q1",methods=["GET","POST"])
def q1():
    r = model.generate_content("How should I diversify my investment portfolio?")
    return(render_template("q1_reply.html",r=r))

@app.route("/q2",methods=["GET","POST"])
def q2():
    q = request.form.get("q")
    r = model.generate_content(q)
    return(render_template("q2_reply.html",r=r))

if __name__ == "__main__":
    app.run()
