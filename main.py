from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import subprocess as sp
import csv
import flask_mysqldb
import sqlite3
#import baixadedados
import mysql
import tratable



app = Flask('__name__')
#abrir dados
arq = open('dadosfim.csv', 'r')

dados = []
dados1 =[]

for line in arq:
    dados.append(line[:-1].split(' ', 17))

    line2 = line[:-1].split(' ', 17)
    dados1.append(line2)


ab = pd.DataFrame(dados1)
print(ab)

#tabela
conn = sqlite3.connect('tabela001.db')
#mycursor = mydb.cursor()
#mycursor.execute("SHOW DATABASES")
#sqlite
#df.to_sql(name='eventos1',con = conn)

#app
@app.route("/")
def homepage():
  return render_template("homepage.html")

@app.route("/all")
def all():
  return render_template("all.html")

@app.route("/info")
def infor():
    return render_template("info.html")

@app.route("/satelite")
def satelite():
    return render_template("satelite.html")

@app.route("/teste")
def s():
    return render_template('teste.html')

@app.route("/eventos")
def teste():
    return render_template('eventos.html')

@app.route("/19981999")
def ano():
    return render_template('1996_1999.html')

@app.route("/2000")
def ano1():
    return render_template('2000.html')

@app.route("/2001")
def ano2():
    return render_template('2001.html')

@app.route("/2002")
def ano3():
    return render_template('2002.html')

@app.route("/2003")
def ano4():
    return render_template('2003.html')

@app.route("/2004")
def ano5():
    return render_template('2004.html')

@app.route("/2005")
def ano6():
    return render_template('2005.html')

@app.route("/2006")
def ano7():
    return render_template('2006.html')

@app.route("/2007")
def ano8():
    return render_template('2007.html')

@app.route("/2008")
def ano9():
    return render_template('2008.html')

@app.route("/2009")
def ano10():
    return render_template('2009.html')

@app.route("/2010")
def ano11():
    return render_template('2010.html')

@app.route("/2011")
def ano12():
    return render_template('2011.html')

@app.route("/2012")
def ano13():
    return render_template('2012.html')

@app.route("/2013")
def ano14():
    return render_template('2013.html')

@app.route("/2014")
def ano15():
    return render_template('2014.html')

@app.route("/2015")
def ano16():
    return render_template('2015.html')

@app.route("/2016")
def ano17():
    return render_template('2016.html')

@app.route("/2017")
def ano18():
    return render_template('2017.html')

@app.route("/2018")
def ano19():
    return render_template('2018.html')

@app.route("/2019")
def ano20():
    return render_template('2019.html')

@app.route("/2020")
def ano21():
    return render_template('2020.html')

@app.route("/2021")
def ano22():
    return render_template('2021.html')

@app.route("/2022")
def ano23():
    return render_template('2022.html')

if __name__ == '__main__':
    app.run(debug=True)


conn.close()

