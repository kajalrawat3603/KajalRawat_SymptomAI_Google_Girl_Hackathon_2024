from datetime import timedelta
from flask import Flask
from flask_dropzone import Dropzone
import os

app=Flask(__name__, template_folder='templates')

app.config.from_object(__name__)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2.0)
app.config['SESSION_PERMANENT'] = True
dir_path=os.path.dirname(os.path.realpath(__file__))

dropzone = Dropzone(app)
from application import routes

