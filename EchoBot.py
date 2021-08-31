import telegram
import os

token = "1910596977:AAFMPBYTwTXXJXqwkXfW5qgVk32aLmfOtRU"
bot = telegram.Bot(token=token)


def SendMsgToTelegram(msg, Toid='388707586'):
    try:
        bot.send_message(chat_id=Toid, text=msg)
        return True
    except Exception as error:
        print(error)
        return False


def SendPlotToTelegram(plt, Toid='388707586'):
    try:
        tmpPath = 'tmp.png'
        plt.savefig(tmpPath)
        bot.send_photo(chat_id=Toid, photo=open(tmpPath, 'rb'))
        os.remove(tmpPath)
        return True
    except Exception as error:
        print(error)
        return False
