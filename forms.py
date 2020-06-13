from wtforms import StringField, validators, Form
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired


class AddPersonForm(Form):
    first_name = StringField('first_name', [validators.InputRequired()])
    second_name = StringField('second_name', [validators.InputRequired()])
    photo = FileField('photo')