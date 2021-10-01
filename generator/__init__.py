# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
migrate = Migrate()

def create_app():
	app = Flask(__name__)
	app.config.from_envvar('APP_CONFIG_FILE')

	#db
	db.init_app(app)
	migrate.init_app(app, db)
	from . import models
	#migrate 객체가 models.py 참조

	#블루프린트(라우트 역할)
	from .views import main_views
	app.register_blueprint(main_views.bp)

	return app