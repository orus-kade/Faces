from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
import forms
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///face.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)



class Embeddings(db.Model):
    __tablename__ = 'embeddings'
    __table_args__ = { 'extend_existing': True }
    # LOC_CODE = db.Column(db.Integer, primary_key=True)
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    first_name = db.Column(db.String(100), nullable=False)
    second_name = db.Column(db.String(100), nullable=False)
    # person_embedding = db.Column(db.Text, nullable=False)
    # data_add = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Embeddings %r>' % self.id

@app.route('/')
@app.route('/home')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/person/add', methods=['POST', 'GET'])
def add_person():
    form = forms.AddPersonForm(request.form, meta={'csrf': False})
    if request.method == "POST" and form.validate():
        first_name = request.form['first_name']
        second_name = request.form['second_name']
        file_storage = request.files['photo']
        file_storage.save('temp_photos/{}'.format(file_storage.filename))
        file_storage.close()
        embedding = Embeddings(first_name=first_name, second_name=second_name)
        try:
            db.session.add(embedding)
            db.session.commit()
            return redirect("/person/saved")
        except:
            return "При добавлении человека произошла ошибка"
    else:
        return render_template("add_person.html", form=form)

@app.route('/person/saved')
def saved_persons():
    deleted = request.args.get('deleted', default=False, type=bool)
    persons = Embeddings.query.order_by(Embeddings.second_name).all()
    return render_template("saved_persons.html", persons=persons, deleted=deleted)

@app.route('/person/delete/<int:id>')
def delete_person(id):
    person = Embeddings.query.get_or_404(id)
    try:
        db.session.delete(person)
        db.session.commit()
        return redirect("/person/saved?deleted=True")
    except:
        return "При удалении произошла ошибка!"

@app.route('/person/edit/<int:id>', methods=['POST', 'GET'])
def edit_person(id):
    person = Embeddings.query.get_or_404(id)
    if request.method == "POST":
        person.first_name = request.form['first_name']
        person.second_name = request.form['second_name']
        try:
            db.session.commit()
            return render_template("edit_person.html", person=person, edited=True)
        except:
            return "При редактировании данных произошла ошибка"
    else:
        return render_template("edit_person.html", person=person, edited=False)

# @app.route('/user/<string:name>/<int:id>')
# def user(name, id):
#     return "User page " + name + " - " + str(id)


if __name__ == "__main__":
    app.run(debug=True)
