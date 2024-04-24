from flask import Flask
from flask_dropzone import Dropzone
import os


app=Flask(__name__)


app.config.from_object(__name__)

dir_path=os.path.dirname(os.path.realpath(__file__))


dropzone = Dropzone(app)
from application import routes

