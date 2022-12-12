""" Specifies routing for the application"""
from flask import render_template, request, jsonify
from app import app
from app import model as model_helper


@app.route("/")
def homepage():
    """ returns rendered homepage """
    # items = db_helper.fetch_todo()
    items = []
    return render_template("index.html", items=items)


@app.route("/m1")
def model1():
    # parse resume from user into string form
    # predict user's job from resume then reccomend options
    items = []
    return render_template("m1.html", items=items)

@app.route("/m2")
def model2():
    # parse resume from user into string form
    # predict user's job from resume then reccomend options
    items = []
    return render_template("m2.html", items=items)

@app.route("/predictjobs", methods=['POST'])
def predictjobs():
    resume_str = request.form['ResumeStr']    
    resume_str = model_helper.clean_resume(resume_str)
    job_prediction = model_helper.predict_job_title_knn(resume_str)
    job_lst = model_helper.reccomend_jobs(job_prediction)    
    return render_template("m1.html", items=job_lst, job_pred=job_prediction)
    
    
@app.route("/measuresim", methods=['POST'])
def measure_similarity():
    resume_str = request.form['ResumeStr']
    resume_str = model_helper.clean_resume(resume_str)
    job_str = request.form['JobDescrStr']
    job_str = model_helper.clean_resume(job_str)

    if resume_str != None and job_str != None:
        resume_pred = model_helper.predict_job_title_knn(resume_str)
        res2_pred = model_helper.predict_job_title_mb(resume_str)
        job_pred = model_helper.classify_job(job_str)
    return render_template("m2.html", res=resume_pred, res2=res2_pred, job=job_pred[0])