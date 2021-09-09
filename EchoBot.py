import telegram
import os

token = "Your Token"
bot = telegram.Bot(token=token)


def SendMsgToTelegram(msg, Toid='Your ID'):
    try:
        bot.send_message(chat_id=Toid, text=msg)
        return True
    except Exception as error:
        print(error)
        return False


def SendPlotToTelegram(plt, Toid='Your ID'):
    try:
        tmpPath = 'tmp.png'
        plt.savefig(tmpPath)
        bot.send_photo(chat_id=Toid, photo=open(tmpPath, 'rb'))
        os.remove(tmpPath)
        return True
    except Exception as error:
        print(error)
        return False
