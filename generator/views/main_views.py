from flask import Blueprint
from flask import Flask, render_template, request, redirect, url_for
import sys
import os
from numpy import random
from generator import db, models
from generator.views.FaceMorph import faceMorph as FM

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/', methods=['GET', 'POST'])
def info():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        work_title = request.form['work_title']
        character_name = request.form['character_name']
        return redirect(url_for('main.results', work_title=work_title, character_name=character_name)) #url_for 함수 이름 받음

@bp.route('/results_<work_title>_<character_name>')
def results(work_title, character_name):
    result = 0
    result_exist = 0
    tmp = models.character.query.filter((models.character.novel == work_title) \
                                                & (models.character.character == character_name)).all()
    result = tmp
    if result:
        result = tmp[0]
        result_exist = 1
        result_gender = result.gender
        result_feature_tmp = result.feature
        result_feature = result_feature_tmp.split('/')
        FM.main(result_gender, result_feature)
        return render_template('results.html', result_exist=result_exist, work_title=work_title, character_name=character_name)
    else:
        return render_template('results.html', result_exist=result_exist)

@bp.route('/introduction')
def introduction():
    return render_template('introduction.html')