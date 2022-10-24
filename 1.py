from flask import Flask, render_template


app = Flask('__name__')
@app.route("/")
def homepage():
   return render_template("homepage.html")

#@app.route("/eventos")
#def planilha():


if __name__ == '__main__':
    app.run(debug=True)

    # cursor.execute(
    #           """
    #          INSERT INTO eventos (id,tipo, ano, data_inicio, data_fim, horario_inicio, horario_fim, angulos, matching_mva, matching_axi, eixo, razao, metrica, angulo_x)
    #         VALUES (?, ?,?, ?,?, ?, ?, ?,?,?, ?, ?, ?)
    #    """, tabela)