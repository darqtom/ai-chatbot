from flask import Flask, render_template, request

from app.chat_bot import ChatBot

BOT_NAME = 'CONRAD'

app = Flask(__name__)
chat_bot = ChatBot(BOT_NAME, .90)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


@app.route('/app', methods=['GET', 'POST'])
def start_chat():
    return render_template('chat.html')


@app.route('/get', methods=['GET', 'POST'])
def get_bot_response():
    data = request.get_json()
    if data:
        if 'userSentence' in data:
            user_message = data['userSentence']
            bot_message = chat_bot.provide_answer(user_message)
            return {'botMessage': bot_message}
        else:
            return {'error': 'Something went wrong!'}
