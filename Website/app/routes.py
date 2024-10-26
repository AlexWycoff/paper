from flask import render_template, flash, redirect, url_for, request
from app import app
from app.forms import LoginForm
from functions import research_query

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    data = request.get_json()
    message = data['message']
    return research_query(message), {'Content-Type': 'text/plain'}

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(
            form.username.data, form.remember_me.data))
        return redirect(url_for('home'))
    return render_template('login.html', form=form)