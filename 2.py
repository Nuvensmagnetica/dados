from flask import Flask, render_template
import pandas as pd
import csv
import sqlite3



with open('Planilhaeventos.csv', encoding='utf-8') as filename:
    df = csv.reader(filename, delimiter=';')
    for l in df:
        id = l[0]
        tipo = l[1]
        ano = l[2]
        data_inicio = l[3]
        data_fim = l[4]
        horario_inicio = l[5]
        horario_fim = l[6]
        angulos = l[7]
        matching_mva = l[8]
        matching_axi = l[9]
        eixo = l[10]
        razao = l[11]
        metrica = l[12]
        angulo_x = l[13]
        print(l)
