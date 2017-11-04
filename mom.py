import Tkinter as tk
import os
import random
import tkMessageBox



def chill():
    root = tk.Tk()
    root.withdraw()
    messages = ["You seem stressed. Maybe you should go outside.",
            "Stress isn't good for the heart. Go eat soomething.",
            "Why don't you watch some Law and Order for a little bit?",
            "Try taking a walk and come back to this later",
            "Do some push ups! Get that blood flowing!",
            "Go have a snack; your work will be there when you get back",
            "Take a nap; you can finish when you wake up",
            "You've been working very hard. You deserve to go play"]
    tkMessageBox.showinfo(title="It's time to take a break", message = random.choice(messages))
    root.destroy()

if __name__ == "__main__":
    chill()
    os.system("pause")
    chill()
