# -*- coding: utf-8 -*-
from flask import Flask

app=Flask("First app")

@app.route('/')
def index():
    return "Yess"
if __name__=="__main__":
    app.run()
