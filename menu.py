# Bibliotecas criadas nesse projeto
from tkcalendar import DateEntry  # Adaptada por Matheus (Prof. Valdir)
import main0
import main1
import main2
import funcoes

# Menu Bar: https://cadernodelaboratorio.com.br/2019/05/14/menu-basico-com-o-sprintpy/
import tkinter.messagebox as tkmsg
import os
from pathlib import Path
from os.path import join, dirname
import tkinter as tk
import gettext
import locale
import configparser
from tkinter import ttk
import sys
import tkinter.filedialog
import logging
# from PIL import ImageTk, Image
from tkinter import font, Entry
from sys import platform
from tkinter import *
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import matplotlib.pyplot as plt

_ = gettext.gettext

global sinalBA
global sinalBT
global decision
sinalBA = 1  # default
sinalBT = 1  # default
decision = 0  # decisao nao salva


class myApp():

    def __init__(self, **kw):
        self.root = Tk()
        self.root.title('Nosso programa é o máximo')
        self.root.state("zoomed")

        self.filename = ''
        nb = ttk.Notebook(self.root)

        self.page1 = Frame(self.root)  # ,bg='gray')
        self.frame_caminho = Frame(self.page1, bd=2)
        self.frame_caminho.pack(side='left', fill='both')
        # Label(self.frame_caminho, text='Caminho').pack(side='left')
        Button(self.frame_caminho, text='Select file', command=self.select_file).pack(expand=True, fill='both')
        self.frame_evento = Frame(self.page1, bd=3)
        self.frame_evento.pack(side='left', fill='both')
        self.frame_inicio = Frame(self.frame_evento)
        self.frame_inicio.pack(fill='both', expand=1)
        Label(self.frame_inicio, text="Start Date", width=10).pack(side='left')
        self.Data_inicio = StringVar(self.root)
        self.cal_inicio = DateEntry(self.frame_inicio, textvariable=self.Data_inicio, borderwidth=2)
        self.cal_inicio.pack(side='left', fill='both', expand=1)
        Label(self.frame_inicio, text='Start Hour', width=10).pack(side='left')
        self.Entry_hora_inicio = Entry(self.frame_inicio, width=5)
        self.Entry_hora_inicio.pack(side='right')
        self.frame_fim = Frame(self.frame_evento)
        self.frame_fim.pack(fill='both', expand=1)
        Label(self.frame_fim, text="End Date", width=10).pack(side='left')
        self.Data_fim = StringVar(self.root)
        self.cal_fim = DateEntry(self.frame_fim, textvariable=self.Data_fim, borderwidth=2)
        self.cal_fim.pack(side='left', fill='both', expand=1)
        Label(self.frame_fim, text='End Hour', width=10).pack(side='left')
        self.Entry_hora_fim = Entry(self.frame_fim, width=5)
        self.Entry_hora_fim.pack(side='right')
        self.frame_plot = Frame(self.page1)
        self.frame_plot.pack(side='left', fill='both')
        self.frame_plot2 = Frame(self.page1)
        self.frame_plot2.pack(side='left', fill='both')
        self.frame_decision1 = Frame(self.page1)
        self.frame_decision1.pack(side='left', fill='both')
        self.frame_decision2 = Frame(self.page1)
        self.frame_decision2.pack(side='left', fill='both')
        self.frame_decision3 = Frame(self.page1)
        self.frame_decision3.pack(side='left', fill='both')

        # 1
        # Button(self.frame_plot,text='MagComp 2',command=self.plotar_magcomp2).pack(fill='both')
        Button(self.frame_plot, text='Data Set', command=self.plotar_magcomp2).pack(fill='both')
        # 2
        # Button(self.frame_plot,text='Max/Min Plot',command=self.plotar_max).pack(fill='both')
        Button(self.frame_plot, text='MVA Hodograms', command=self.plotar_max).pack(fill='both')
        # 3
        # Button(self.frame_plot2,text='MagComp Plot',command=self.plotar_magcomp).pack(fill='both')
        Button(self.frame_plot2, text='Axes Directions', command=self.plotar_magcomp).pack(fill='both')
        # 4
        # Button(self.frame_plot2,text='MVA info',command=self.plotar_MVAinfo).pack(fill='both')
        Button(self.frame_plot2, text='MVA Results', command=self.plotar_MVAinfo).pack(fill='both')

        #        Button(self.frame_plot3,text='Helix plot',command=self.plotar_Helix).pack(fill='both')

        ## Apagar o arquivo 'hodograms.txt' (se ele existe) sempre que abrir
        if os.path.isfile('hodograms.txt'):
            os.remove('hodograms.txt')

        ##IDEAL##
        self.page2 = Frame(self.root)  # ,bg='red')
        ########
        # self.frame_entrada = Frame(self.page2,bd=2)
        # self.frame_entrada.pack(side='left',fill='both')
        self.frame_phi = Frame(self.page2, bd=2)  # Frame(self.frame_entrada)
        self.frame_phi.pack(side='left', fill='both')  # (fill='both',expand=1)
        # Label(self.frame_phi,text='Angulo Phi (em graus)',width=10).pack(side='left')
        Label(self.frame_phi, text='Longitudinal Angle (in degree)', width=25).pack(side='left')
        self.Entry_ang_phi = Entry(self.frame_phi, width=5)
        self.Entry_ang_phi.pack(side='right')

        self.frame_theta = Frame(self.page2, bd=3)  # Frame(self.frame_entrada)
        self.frame_theta.pack(side='left', fill='both')  # (fill='both',expand=1)
        # Label(self.frame_theta,text='Angulo Theta (em graus)',width=10).pack(side='left')
        Label(self.frame_theta, text='Latitudinal Angle (in degree)', width=25).pack(side='left')
        self.Entry_ang_theta = Entry(self.frame_theta, width=5)
        self.Entry_ang_theta.pack(side='right')

        self.frame_helicity = Frame(self.page2, bd=4)  # Frame(self.frame_entrada)
        self.frame_helicity.pack(side='left', fill='both')  # (fill='both',expand=1)
        Label(self.frame_helicity, text='Helicity', width=10).pack(side='left')
        self.Entry_helicity = Entry(self.frame_helicity, width=5)
        self.Entry_helicity.pack(side='right')

        self.frame_plot = Frame(self.page2)
        self.frame_plot.pack(side='left', fill='both')
        Button(self.frame_plot, text='Hodograms', command=self.plotar_ideal).pack(fill='both')

        Button(self.frame_plot, text='Helix plot', command=self.plotar_Helix).pack(fill='both')

        ########
        self.page3 = Frame(self.root)  # ,bg='blue')
        self.frame_reference = Frame(self.page3, bd=2)
        self.frame_reference.pack(fill='both')
        Label(self.frame_reference, text='In construction').pack()

        nb.add(self.page1, text="REAL")
        nb.add(self.page2, text="MODEL")
        nb.add(self.page3, text="ABOUT")

        nb.pack(side='top')

        self.frame_graph = Frame(self.root, bg='blue')
        self.frame_graph.pack(expand=1, side='bottom', fill='both')

        self.img, self.axes = plt.subplots()

        self.canvas_img = FigureCanvasTkAgg(self.img, self.frame_graph)
        self.toolbar = NavigationToolbar2Tk(self.canvas_img, self.frame_graph)
        self.canvas_img.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas_img._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.img.clf()

    def select_file(self):
        # self.file = askdirectory(initialdir="c:/",title = "Selecionar Arquivo",parent=self.root)
        self.filename = askopenfilename(filetypes=(("Texto", "*.txt"),), parent=self.root)

    def plotar_ideal(self):

        # Entry verify
        if error_ideal(self.Entry_ang_phi.get(), self.Entry_ang_theta.get(), self.Entry_helicity.get()):

            # ang_phi = int(self.Entry_ang_phi.get())
            ang_theta = int(self.Entry_ang_phi.get())
            # ang_theta = int(self.Entry_ang_theta.get())
            ang_phi = int(self.Entry_ang_theta.get())
            helicity = int(self.Entry_helicity.get())
            if helicity == 1:
                hand = 'l'
            if helicity == -1:
                hand = 'r'

            plt.clf()

            [figuraideal, error_msg] = main2.main2(ang_theta, ang_phi, hand)
            if not figuraideal:
                tkmsg.showerror("Error", error_msg)
            else:
                # plt.savefig('FIGURAidealteste.pdf')
                self.canvas_img.draw()

    def plotar_max(self):
        # texto no botão: 'MVA Hodograms'

        # Limpa botoes de decisão se existir
        self.frame_decision1.destroy()
        self.frame_decision2.destroy()
        self.frame_decision3.destroy()

        # Criando Botoes de Decisão
        self.frame_decision1 = Frame(self.page1)
        self.frame_decision1.pack(side='left', fill='both')
        self.frame_decision2 = Frame(self.page1)
        self.frame_decision2.pack(side='left', fill='both')
        self.frame_decision3 = Frame(self.page1)
        self.frame_decision3.pack(side='left', fill='both')

        Button(self.frame_decision1, text='-BA', command=self.plotar_menosBA).pack(fill='both')
        Button(self.frame_decision1, text='-BT', command=self.plotar_menosBT).pack(fill='both')
        Button(self.frame_decision2, text='-BA X -BT', command=self.plotar_menosBAxBT).pack(fill='both')
        Button(self.frame_decision2, text='Default', command=self.plotar_default).pack(fill='both')
        Button(self.frame_decision3, text='Save Decision', command=self.save_decision).pack(fill='both')

        data_inicio = (self.Data_inicio.get()).split('/')
        data_fim = (self.Data_fim.get()).split('/')
        # Entry verify
        if error_diahora(self.Entry_hora_inicio.get(), self.Entry_hora_fim.get(), data_inicio, data_fim):

            hora_inicio = int(self.Entry_hora_inicio.get())

            hora_fim = int(self.Entry_hora_fim.get())

            inicio_fim = [data_inicio[2], data_inicio[1], data_inicio[0], str(hora_inicio), data_fim[2], data_fim[1],
                          data_fim[0], str(hora_fim)]

            plt.clf()

            if self.filename == '':
                tkmsg.showerror("Error", 'Not selected input file!')
            else:
                print(inicio_fim)
                print(self.filename)
                [FIGURA1, error_msg] = main0.main0(inicio_fim, self.filename, 'plotmaxmin')
                global tempsinalBA
                global tempsinalBT
                tempsinalBA = 1
                tempsinalBT = 1

                if not FIGURA1:
                    tkmsg.showerror("Error", error_msg)
                else:
                    # plt.savefig('FIGURAmaxminteste.png')
                    self.canvas_img.draw()

    def plotar_menosBA(self):
        global tempsinalBA
        global tempsinalBT

        plt.clf()

        [FIGURA1, error_msg] = main0.plot_decision('plotar_menosBA', sinalBA, sinalBT, decision)
        tempsinalBA = -1
        tempsinalBT = 1

        if not FIGURA1:
            tkmsg.showerror("Error", error_msg)
        else:
            # plt.savefig('FIGURAmaxminteste.pdf')
            self.canvas_img.draw()

    def plotar_menosBT(self):
        global tempsinalBA
        global tempsinalBT

        plt.clf()

        [FIGURA1, error_msg] = main0.plot_decision('plotar_menosBT', sinalBA, sinalBT, decision)
        tempsinalBA = 1
        tempsinalBT = -1

        if not FIGURA1:
            tkmsg.showerror("Error", error_msg)
        else:
            # plt.savefig('FIGURAmaxminteste.pdf')
            self.canvas_img.draw()

    def plotar_menosBAxBT(self):
        global tempsinalBT
        global tempsinalBA

        plt.clf()

        [FIGURA1, error_msg] = main0.plot_decision('plotar_menosBAxBT', sinalBA, sinalBT, decision)
        tempsinalBA = -1
        tempsinalBT = -1

        if not FIGURA1:
            tkmsg.showerror("Error", error_msg)
        else:
            # plt.savefig('FIGURAmaxminteste.pdf')
            self.canvas_img.draw()

    def plotar_default(self):
        global tempsinalBT
        global tempsinalBA

        plt.clf()

        [FIGURA1, error_msg] = main0.plot_decision('plotar_default', sinalBA, sinalBT, decision)
        tempsinalBA = 1
        tempsinalBT = 1

        if not FIGURA1:
            tkmsg.showerror("Error", error_msg)
        else:
            # plt.savefig('FIGURAmaxminteste.pdf')
            self.canvas_img.draw()

    def save_decision(self):
        global tempsinalBT
        global tempsinalBA
        global sinalBT
        global sinalBA
        global decision

        sinalBT = tempsinalBT
        sinalBA = tempsinalBA
        decision = 1  # decisao salva

    def plotar_MVAinfo(self):
        # texto no botão: 'MVA Results'
        global sinalBT
        global sinalBA
        global decision

        # print('sinalBA= ', sinalBA)
        # print('sinalBT= ', sinalBT)
        # print('decision= ', decision)

        # Destroy decision botons
        self.frame_decision1.destroy()
        self.frame_decision2.destroy()
        self.frame_decision3.destroy()

        data_inicio = (self.Data_inicio.get()).split('/')
        data_fim = (self.Data_fim.get()).split('/')

        # Entry verify
        if error_diahora(self.Entry_hora_inicio.get(), self.Entry_hora_fim.get(), data_inicio, data_fim):

            hora_inicio = int(self.Entry_hora_inicio.get())

            hora_fim = int(self.Entry_hora_fim.get())

            inicio_fim = [data_inicio[2], data_inicio[1], data_inicio[0], str(hora_inicio), data_fim[2], data_fim[1],
                          data_fim[0], str(hora_fim)]

            plt.clf()

            if self.filename == '':
                tkmsg.showerror("Error", 'Not selected input file!')
            else:

                if verifica_arq(inicio_fim):
                    # print('arquivo com mesmo evento')
                    [FIGURA, error_msg] = main0.plot_decision('plotar_mvaresults', sinalBA, sinalBT, decision)

                else:
                    # print('arquivo com evento diferente')
                    [FIGURA, error_msg] = main0.main0(inicio_fim, self.filename, 'MVAinfo')

                if not FIGURA:
                    tkmsg.showerror("Error", error_msg)
                else:
                    # plt.savefig('FIGURAMVAinfoteste.pdf')
                    self.canvas_img.draw()

    def plotar_magcomp(self):
        # Texto no botão: "Axes Directions"
        global sinalBT
        global sinalBA
        global decision

        # Limpa botoes de decisão se existir
        self.frame_decision1.destroy()
        self.frame_decision2.destroy()
        self.frame_decision3.destroy()

        data_inicio = (self.Data_inicio.get()).split('/')
        data_fim = (self.Data_fim.get()).split('/')
        # Entry verify
        if error_diahora(self.Entry_hora_inicio.get(), self.Entry_hora_fim.get(), data_inicio, data_fim):

            hora_inicio = int(self.Entry_hora_inicio.get())
            hora_fim = int(self.Entry_hora_fim.get())

            inicio_fim = [data_inicio[2], data_inicio[1], data_inicio[0], str(hora_inicio), data_fim[2], data_fim[1],
                          data_fim[0], str(hora_fim)]

            plt.clf()

            if self.filename == '':
                tkmsg.showerror("Error", 'Not selected input file!')
            else:

                #                [FIGURA2,error_msg] = main0.main0(inicio_fim,self.filename,'plotmagcomp')
                if verifica_arq(inicio_fim):
                    print('arquivo com mesmo evento')
                    [FIGURA2, error_msg] = main0.plot_decision('plotar_axisdirections', sinalBA, sinalBT, decision)

                else:
                    print('arquivo com evento diferente')
                    [FIGURA2, error_msg] = main0.main0(inicio_fim, self.filename, 'plotmagcomp')

                if not FIGURA2:
                    tkmsg.showerror("Error", error_msg)
                else:
                    # plt.savefig('FIGURAmagcompteste.pdf')
                    self.canvas_img.draw()

    def plotar_magcomp2(self):
        # Texto no botão: "Data Set"

        # Limpa botoes de decisão se existir
        self.frame_decision1.destroy()
        self.frame_decision2.destroy()
        self.frame_decision3.destroy()

        # tkmsg.showwarning("Attention", "Check that the source file is suitable for this task.")

        data_inicio = (self.Data_inicio.get()).split('/')
        data_fim = (self.Data_fim.get()).split('/')
        # Entry verify
        if error_diahora(self.Entry_hora_inicio.get(), self.Entry_hora_fim.get(), data_inicio, data_fim):

            hora_inicio = int(self.Entry_hora_inicio.get())
            hora_fim = int(self.Entry_hora_fim.get())

            inicio_fim = [data_inicio[2], data_inicio[1], data_inicio[0], str(hora_inicio), data_fim[2], data_fim[1],
                          data_fim[0], str(hora_fim)]

            plt.clf()

            if self.filename == '':
                tkmsg.showerror("Error", 'Not selected input file!')
            else:
                [FIGURA7, error_msg] = main1.main1(inicio_fim, self.filename, 'plotmagcomp2')
                if not FIGURA7:
                    tkmsg.showerror("Error", error_msg)
                else:
                    # plt.savefig('FIGURAmagcompteste2.pdf')
                    self.canvas_img.draw()

    def plotar_Helix(self):
        # Entry verify
        if error_ideal(self.Entry_ang_phi.get(), self.Entry_ang_theta.get(), self.Entry_helicity.get()):

            # ang_phi = int(self.Entry_ang_phi.get())
            ang_theta = int(self.Entry_ang_phi.get())
            # ang_theta = int(self.Entry_ang_theta.get())
            ang_phi = int(self.Entry_ang_theta.get())
            helicity = int(self.Entry_helicity.get())
            if helicity == 1:
                hand = 'l'
            if helicity == -1:
                hand = 'r'

            plt.clf()
            #            plt.figure(numfig, dpi=300, figsize=(14,12))
            FIGURA = funcoes.funcao4(ang_theta * np.pi / 180, ang_phi * np.pi / 180, hand)

            if not FIGURA:
                tkmsg.showerror("Error", 'Not Processing!')
            else:
                # plt.savefig('FIGURAHelixteste.pdf')
                self.canvas_img.draw()

    ##        data_inicio = (self.Data_inicio.get()).split('/')
    ##        data_fim = (self.Data_fim.get()).split('/')
    ##        # Entry verify
    ##        if error_diahora(self.Entry_hora_inicio.get(),self.Entry_hora_fim.get(),data_inicio,data_fim):
    ##
    ##            hora_inicio = int(self.Entry_hora_inicio.get())
    ##
    ##            hora_fim = int(self.Entry_hora_fim.get())
    ##
    ##            inicio_fim = [data_inicio[2],data_inicio[1],data_inicio[0],str(hora_inicio),data_fim[2],data_fim[1],data_fim[0],str(hora_fim)]
    ##
    ##            plt.clf()
    ##
    ##            if self.filename == '':
    ##                tkmsg.showerror("Error", 'Not selected input file!')
    ##            else:
    ##                [FIGURA,error_msg] = main0.main0(inicio_fim,self.filename,'Helix')
    ##
    ##                if not FIGURA:
    ##                    tkmsg.showerror("Error", error_msg)
    ##                else:
    ##                    plt.savefig('FIGURAMVAinfoteste.pdf')
    ##                    self.canvas_img.draw()

    ## destroy usin windows-close botton
    #    self.root.protocol('WM_DELETE_WINDOW', funcaoSair)

    # def funcaoSair(self):
    #    #a = tkmsg.askokcancel('Quit', 'Are you sure you want to exit?')
    #    print('entrei')
    #    #self.root.quit()     # stops mainloop
    #    #self.root.destroy()  # this is necessary on Windows to prevent
    # Fatal Pyt hon Error: PyEval_RestoreThread: NULL tstate

    ##    def funcaoIdeal(self):
    ##        pass
    ##
    ##    def funcaoReal(self):
    ##        pass

    def execute(self):
        self.root.mainloop()


def error_diahora(h0, h1, d0, d1):
    if h0.isdigit() and h1.isdigit():
        h0 = int(h0)
        h1 = int(h1)
        if h0 > 23 or h0 < 0 or h1 > 23 or h1 < 0:
            tkmsg.showerror("Error", "Entry 'hour' incorrect!")
            return False
        for i in range(len(d0)):
            d0[i] = int(d0[i])
            d1[i] = int(d1[i])
        # if  datas iguais e h_f < h_i ou mes/ano iguais e dfinal < dinicial  ou anos iguais e mes_f < mes_i      ou ano_f < ano_i
        if (d1 == d0 and h1 <= h0) or (d1[1:3] == d0[1:3] and d1[0] < d0[0]) or (d1[2] == d0[2] and d1[1] < d0[1]) or (
                d1[2] < d0[2]):
            tkmsg.showerror("Error", "'End' entry less than or equal to 'start'!")
            return False
    else:
        tkmsg.showerror("Error", "Not accept entry 'hour' negative or with string!")
        return False
    return True


def verifica_arq(inicio_fim):
    global sinalBA
    global sinalBT
    global decision

    # inicio_fim = [data_inicio[2],data_inicio[1],data_inicio[0],str(hora_inicio),data_fim[2],data_fim[1],data_fim[0],str(hora_fim)]
    if os.path.isfile('hodograms.txt'):
        # abertura do arquivo
        arq = open('hodograms.txt', 'r')

        # Extrai dados linha a linha do arquivo arqdados
        dados = []
        count = 0
        arqcol = []
        vec_theta = np.zeros(2)
        vec_phi = np.zeros(2)

        for line in arq:

            if count == 1:  # ano_inicio
                count = 0
                line2 = line[:-1].split(' ')
                if not inicio_fim[0] == int(line2[0]):
                    sinalBA = 1  # default
                    sinalBT = 1  # default
                    decision = 0  # decisao nao salva
                    return False

            if count == 2:  # mes_inicio
                count = 0
                line2 = line[:-1].split(' ')
                if not inicio_fim[1] == int(line2[0]):
                    sinalBA = 1  # default
                    sinalBT = 1  # default
                    decision = 0  # decisao nao salva
                    return False

            if count == 3:  # dia_inicio
                count = 0
                line2 = line[:-1].split(' ')
                if not inicio_fim[2] == int(line2[0]):
                    sinalBA = 1  # default
                    sinalBT = 1  # default
                    decision = 0  # decisao nao salva
                    return False

            if count == 4:  # hora_inicio
                count = 0
                line2 = line[:-1].split(' ')
                if not int(inicio_fim[3]) == int(line2[0]):
                    sinalBA = 1  # default
                    sinalBT = 1  # default
                    decision = 0  # decisao nao salva
                    return False

            if count == 5:  # ano_fim
                count = 0
                line2 = line[:-1].split(' ')
                if not inicio_fim[4] == int(line2[0]):
                    sinalBA = 1  # default
                    sinalBT = 1  # default
                    decision = 0  # decisao nao salva
                    return False

            if count == 6:  # mes_fim
                count = 0
                line2 = line[:-1].split(' ')
                if not inicio_fim[5] == int(line2[0]):
                    sinalBA = 1  # default
                    sinalBT = 1  # default
                    decision = 0  # decisao nao salva
                    return False

            if count == 7:  # dia_fim
                count = 0
                line2 = line[:-1].split(' ')
                if not inicio_fim[6] == int(line2[0]):
                    sinalBA = 1  # default
                    sinalBT = 1  # default
                    decision = 0  # decisao nao salva
                    return False

            if count == 8:  # hora_fim
                count = 0
                line2 = line[:-1].split(' ')
                if not int(inicio_fim[7]) == int(line2[0]):
                    sinalBA = 1  # default
                    sinalBT = 1  # default
                    decision = 0  # decisao nao salva
                    return False
            if decision == 0:
                return False

            if line == 'ano_inicio\n':
                count = 1
            elif line == 'mes_inicio\n':
                count = 2
            elif line == 'dia_inicio\n':
                count = 3
            elif line == 'hora_inicio\n':
                count = 4
            elif line == 'ano_fim\n':
                count = 5
            elif line == 'mes_fim\n':
                count = 6
            elif line == 'dia_fim\n':
                count = 7
            elif line == 'hora_fim\n':
                count = 8

        arq.close()
        return True
    else:
        return False
    return True


def error_ideal(phi, theta, helicity):
    if is_float(phi) and is_float(theta):
        phi = float(phi)
        theta = float(theta)

        if phi < 0 or phi > 360:
            tkmsg.showerror("Error", 'The phi angle must be in the range [0°, 360°]')
            return False
        elif abs(theta) > 90:
            tkmsg.showerror("Error", 'The theta angle must be in the range [-90°, 90°]')
            return False
    else:
        tkmsg.showerror("Error", "Not accept entry 'phi' or 'theta' with string!")
        return False

    if is_float(helicity):
        helicity = float(helicity)
        if helicity != -1 and helicity != 1:
            tkmsg.showerror("Error", "Accept 'helicity' input with number 1 or -1! (-1 = right, 1= left)")
            return False
    else:
        tkmsg.showerror("Error", "Accept 'helicity' input with number 1 or -1! (-1 = right, 1= left)")
        return False

    return True


def is_float(val):
    try:
        num = float(val)
    except ValueError:
        return False
    return True


def main(args):
    app_proc = myApp()
    app_proc.execute()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
