# Programa Principal
# Combina as ações dos dois programas, automaticamente. Ou seja:
# Entrada: Arquivo com Dados + Arquivo com MCinicio/fim
# Saídas Programa 1 e Programa 2
#
# Programa 1 - para forcefree + rotação -> opções de Max/Min
# Entrada: Arquivo com um par de ângulos (phi e theta)
#
# Programa 2 - processa MVA+BLatBlon+MCtype+Programa1 -> opções de Max/Min
# Entrada: Arquivo com Dados + Arquivo com MCinicio/fim
#
# iniciado em 09/08/19
#
#######

# Bibliotecas criadas nesse projeto
#from funcao import funcao
import figuras
import funcoes

###

# Bibliotecas próprias Python
import numpy as np
import pandas as pd
# import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as ss


# VERIFICAR SE SÃO NECESSÁRIAS
# import scipy.stats as st #import linregress
# import xlwt #para gerar arquivo .xls
# import time
# import math #para fatorial, pi, cos, sin
###


############################################################################
################################   ##   ####################################
################################# #### #####################################
################################# #### #####################################
################################   ##   ####################################
############################################################################

# Entrada: Arquivo com os dados e arquivo com inicio e fim da nuvem

# NOMEARQ = "ACE_2002_110-111" #CASO SEN
# NOMEARQ = "ACE_2002_273-274" #CASO NES
# NOMEARQ = "ACE_2003_324-325" #CASO ESW
# NOMEARQ = "ACE_1998_232-233" #CASO SWN
# NOMEARQ = "ACE_2001_61-87" #CASO WSE 03/2001
# NOMEARQ = "ACE_2000_52-53" #CASO WNE
# NOMEARQ = "ACE_1999_221-222" #CASO ENW
# NOMEARQ = "ACE_1999_49-50" #CASO NWS
arqdados=open('2004.txt')
arq2 = arqdados.readlines()
dados = []
inicio = 0
arqcol = []
#for linha in linhas:
	#print(linha)

#def main0(table, arqdados, opt_plot):
    # abertura do arquivo
    #arq = open(arqdados, 'r')
    #arq2 = arq.readlines()
    # Extrai dados linha a linha do arquivo arqdados
for line in arq2:
    if inicio == 1:  # Pular cabeçalho
            line2 = line[:-1].split(' ')
            while '' in line2:
                del line2[line2.index('')]  # filtro de espaços

            dados.append(line2)  # Copia dados em kip separando colunas
    elif line != "BEGIN DATA":
            arqcols = line[:-1].split(' ')
    if line == "BEGIN DATA":
            inicio = 1

    Bmagcalc = 0

    # Identificação de Colunas e mensagem ERROR
    arqcols = np.array(arqcols)

    #    elif not ('min' in arqcols):
    #        return['',"Input file without 'min' data!"]
    #    elif not ('sec' in arqcols):
    #        return['',"Input file without 'sec' data!"]

    for i in range(len(arqcols)):
        if arqcols[i] == 'year':
            yearcol = i
        elif arqcols[i] == 'day':
            daycol = i
        elif arqcols[i] == 'hr':
            hrcol = i
        elif arqcols[i] == 'min':
            mincol = i
        elif arqcols[i] == 'sec':
            seccol = i
        elif arqcols[i] == 'Bmag':
            Bmagcol = i
        elif arqcols[i] == 'Bgse_x' or arqcols[i] == 'B_gse_x':
            Bxcol = i
        elif arqcols[i] == 'Bgse_y' or arqcols[i] == 'B_gse_y':
            Bycol = i
        elif arqcols[i] == 'Bgse_z' or arqcols[i] == 'B_gse_z':
            Bzcol = i

    # Matriz para guardar resultados:
    MC = []  # lista vazia

    # Processamento linha a linha do arquivo TABLE
    # for tablex in range((len(table))):
    lendados = len(dados) - 1
    print('lendados= ', lendados)
    start_year = float(table[0])
    start_day = funcoes.doy(float(table[0]), float(table[1]), float(table[2]))
    start_hour = float(table[3])
    end_year = float(table[4])
    end_day = funcoes.doy(float(table[4]), float(table[5]), float(table[6]))
    end_hour = float(table[7])

    if start_hour == 24:  # 24h = 0h do dia seguinte
        start_hour = 0
        start_day = start_day + 1

    if end_hour == 24:  # 24h = 0h do dia seguinte
        end_hour = 0
        end_day = start_day + 1

    # Localiza o inicio do evento (inicia na linha x_begin)
    x_begin = 0

    # Erro "data de entrada inferior à primeira data do arquivo"

    test = float(dados[x_begin][yearcol])  # year

    while test < start_year:
        x_begin = x_begin + 1

    test = float(dados[x_begin][daycol])  # day

    while test < start_day:
        x_begin = x_begin + 1


    test = float(dados[x_begin][hrcol])  # hr

    while test < start_hour:
        x_begin = x_begin + 1


    print('inicio= ' + str(dados[x_begin][yearcol]) + str(dados[x_begin][daycol]) + str(dados[x_begin][hrcol]))
    # Localiza o fim do evento (termina na linha x_end)
    x_end = x_begin + 1

    test = float(dados[x_end][yearcol])  # year

    while test < end_year:
        x_end = x_end + 1
        test = float(dados[x_end][yearcol])  # year

    test = float(dados[x_end][daycol])  # day

    while test < end_day:
        x_end = x_end + 1
        test = float(dados[x_end][daycol])  # day

    test = float(dados[x_end][hrcol])  # hr

    while test < end_hour:
        x_end = x_end + 1

        test = float(dados[x_end][hrcol])  # hr

    print('fim= ' + str(dados[x_end][yearcol]) + str(dados[x_end][daycol]) + str(dados[x_end][hrcol]))

    # Inicializa vetores
    Bx = np.zeros(x_end - x_begin + 1)  # componente x do campo
    By = np.zeros(x_end - x_begin + 1)  # componente y do campo
    Bz = np.zeros(x_end - x_begin + 1)  # componente z do campo
    modB = np.zeros(x_end - x_begin + 1)  # Magnitude do campo
    hr = np.zeros(x_end - x_begin + 1)  # hora

    # Carregamento linha a linha do arquivo DADOS
    cont = -1
    for x in range(x_begin, x_end + 1):
        cont = cont + 1
        Bx[cont] = float(dados[x][Bxcol])  # Bx
        By[cont] = float(dados[x][Bycol])  # By
        Bz[cont] = float(dados[x][Bzcol])  # Bz
        if Bmagcalc == 1:
            modB[cont] = np.sqrt(Bx[cont] ** 2 + By[cont] ** 2 + Bz[cont] ** 2)
        else:
            modB[cont] = float(dados[x][Bmagcol])  # Bmag
        hr[cont] = float(dados[x][hrcol])  # hr
    # PROCESSAMENTO DE UMA NUVEM

    ######################
    ######## MVA #########
    ######################

    BL, BM, BN, eigval, eigvec, phi, theta, chi = funcoes.funcao2(Bx, By, Bz)
    theta_mva = theta  # em graus - para plot
    phi_mva = phi  # em graus - para plot

    theta = round(theta) * np.pi / 180
    phi = round(phi) * np.pi / 180

    ######################
    ###### Métrica #######
    ######################

    Pmetric = funcoes.metric(eigval)

    ######################
    ##### BLAT/BLONG #####
    ######################

    BLat, BLon = funcoes.funcao3(Bx, By, Bz)  # calculo a partir dos dados originais
    # BLat, BLon = funcoes.funcao3(BN, BM, BL) #teste calculo a partir do resultado MVA -> [Bx*,By*,Bz*]=[BR,BA,BT]=[BN,BM,BL]

    # meire = funcoes.filtro_media(BLat)

    ########################################
    ##### Filtro de Sinal e Id. MCType #####
    ########################################

    kernel_size = round(len(BLat) / 10 - 1)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size - 1
    fBLat = ss.medfilt(BLat, kernel_size)
    fBLon = ss.medfilt(BLon, kernel_size)
    tipo, hand = funcoes.funcaotype(fBLat, fBLon, theta)

    ########################################
    ######### Gráficos Artigo ##############
    ########################################
    r = int(len(modB) / 50) - 1  # rescaling
    ##    plt.close('all')
    ##    numfig=1
    ##    plt.figure(numfig, dpi=300, figsize=(7,6))
    ##    FIGURA1 = figuras.figura1(modB[::r],Bx[::r],By[::r],Bz[::r],BLat[::r],BLon[::r],'')
    ##    plt.savefig('Fig1'+tipo+'-theta'+str(int(theta*180/np.pi))+'-phi'+str(int(phi*180/np.pi))+hand+'H_real.png')
    ###    plt.show()
    ##    plt.close('all')
    ##    numfig=numfig+1
    ##    plt.figure(numfig, dpi=300, figsize=(7,6))
    ##    FIGURA3 = figuras.figura2(Bx[::r],By[::r],Bz[::r],'')
    ##
    ##    plt.savefig('Fig2'+tipo+'-theta'+str(int(theta*180/np.pi))+'-phi'+str(int(phi*180/np.pi))+hand+'H_real.png')
    ##    plt.close('all')
    ##
    # Variáveis para plot junto real e ideal
    pltmodB_real = modB[::r]
    pltBx_real = Bx[::r]
    pltBy_real = By[::r]
    pltBz_real = Bz[::r]
    pltBL = BL[::r]
    pltBM = BM[::r]
    pltBN = BN[::r]
    pltBlat_real = BLat[::r]
    pltBlon_real = BLon[::r]

    ############################################################################
    ###################################   ######################################
    #################################### #######################################
    #################################### #######################################
    ###################################   ######################################
    ############################################################################

    vec_phi, vec_theta, vec_hand = funcoes.anglelist(round(phi * 180 / np.pi), round(theta * 180 / np.pi), tipo)

    pltmodB_id = []
    pltBx_id = []
    pltBy_id = []
    pltBz_id = []
    pltBlat_id = []
    pltBlon_id = []

    for x in range(len(vec_theta)):
        phi = vec_phi[x]  # em graus
        theta = vec_theta[x]  # em graus
        hand = vec_hand[x]

        ###################################################
        ##### Relação sistema GSM - Sistema Polar yOz #####
        ###################################################

        omega = funcoes.funcao0(phi, theta)

        ######################
        ##### FORCE-FREE #####
        ######################

        # Parametros iniciais para Coeficientes de Bessel
        B0 = 40
        alpha = 1
        #####################################################
        # até 15-04-20 estávamos usando a helicidade assim ##
        # if (hand == 'l' or hand == 'L' or hand == 'left' or hand == 'LEFT' or hand == 'Left'):
        #    H = 1
        # else:
        #        H = -1
        #
        # agora, vamos calcula-la baseado em Huttunen2005  ##
        # e Lynch2003                                      ##
        if (hand == 'l' or hand == 'L' or hand == 'left' or hand == 'LEFT' or hand == 'Left'):
            if phi < 180:
                H = 1  # -1
            else:
                H = -1  # 1
        else:
            if phi < 180:
                H = -1  # 1
            else:
                H = 1  # -1
        ##################################################

        r_max = 10

        r, idBA, idBT, idBR, modB, ROT, Bx, By, Bz = funcoes.funcao1(B0, alpha, H, r_max, omega)

        ######################
        ######## MVA #########
        ######################

        # Parâmetros que limitam os dados utilizados no MVA
        ##nesse caso, são os valores usados no modelo
        MC_inicio = 75
        MC_fim = 126

        # BL, BM, BN, eigval, eigvec = funcoes.funcao2(Bx[MC_inicio:MC_fim], By[MC_inicio:MC_fim], Bz[MC_inicio:MC_fim])

        ######################
        ##### BLAT/BLONG #####
        ######################

        idBLat, idBLon = funcoes.funcao3(idBR[MC_inicio:MC_fim], idBA[MC_inicio:MC_fim], idBT[MC_inicio:MC_fim])

        #############################
        ##### MCtype automático #####
        #############################

        idtipo, idhand = funcoes.funcaotype(idBLat, idBLon, phi)

        ######################
        ###### TUBO 3D #######
        ######################

        ##        numfig=1
        ##        plt.figure(numfig)#, figsize=(17,8))
        ##        FIGURA1 = funcoes.funcao4(theta,phi,hand,omega)
        ##        plt.savefig(tipo+'tubo'+'-theta'+str(int(theta*180/np.pi))+'-phi'+str(int(phi*180/np.pi))+hand+'H.png')
        ##    #    plt.show()
        ##
        ##        numfig=numfig+1
        ##        plt.figure(numfig, dpi=300, figsize=(7,6))
        ##        FIGURA2 = figuras.figura1(modB[MC_inicio:MC_fim],Bx[MC_inicio:MC_fim],By[MC_inicio:MC_fim],Bz[MC_inicio:MC_fim],idBLat,idBLon,'')
        ##        plt.savefig('Fig1'+tipo+'-theta'+str(int(theta*180/np.pi))+'-phi'+str(int(phi*180/np.pi))+hand+'H.png')
        ##    #    plt.show()
        ##        numfig=numfig+1
        ##        plt.figure(numfig, dpi=300, figsize=(7,6))
        ##        FIGURA3 = figuras.figura2(Bx[MC_inicio:MC_fim],By[MC_inicio:MC_fim],Bz[MC_inicio:MC_fim],'')
        ##        plt.savefig('Fig2'+tipo+'-theta'+str(int(theta*180/np.pi))+'-phi'+str(int(phi*180/np.pi))+hand+'H.png')
        ##        plt.close('all')

        # Variáveis para plot junto real e ideal
        pltmodB_id.append(modB[MC_inicio:MC_fim])
        pltBx_id.append(Bx[MC_inicio:MC_fim])
        pltBy_id.append(By[MC_inicio:MC_fim])
        pltBz_id.append(Bz[MC_inicio:MC_fim])
        pltBlat_id.append(idBLat)
        pltBlon_id.append(idBLon)

    # SINAIS --> PLOT JUNTOS
    numfig = 1
    if opt_plot == 'plotmagcomp':
        plt.figure(numfig, dpi=300, figsize=(14, 12))
        FIGURA3 = figuras.figura3(pltmodB_real, pltBx_real, pltBy_real, pltBz_real, pltBlat_real, pltBlon_real,
                                  pltmodB_id[0], pltBx_id[0], pltBy_id[0], pltBz_id[0], pltBlat_id[0], pltBlon_id[0],
                                  pltmodB_id[1], pltBx_id[1], pltBy_id[1], pltBz_id[1], pltBlat_id[1], pltBlon_id[1],
                                  vec_theta, vec_phi, phi_mva, theta_mva, '')
        # plt.close('all')
        # plt.savefig('Fig3-plotmagcomp.pdf')
        print(FIGURA3)
    # plt.show()
    # plt.get_current_fig_manager().window.state('zoomed')
    ##    plt.savefig('Fig3-'+NOMEARQ+'-'+tipo+'-'+hand+'H_juntos.pdf')

    if opt_plot == 'plotmaxmin':
        plt.figure(numfig, dpi=300, figsize=(14, 12))  # troquei Bx, By Bz por BN BM BL
        FIGURA4 = figuras.figura4(pltBN, pltBM, pltBL,
                                  pltBx_id[0], pltBy_id[0], pltBz_id[0],
                                  pltBx_id[1], pltBy_id[1], pltBz_id[1],
                                  vec_theta, vec_phi, theta_mva, phi_mva, '')
        ##    plt.savefig('Fig4-'+NOMEARQ+'-'+tipo+'-'+hand+'H_juntos.pdf')
        # plt.close('all')
        # plt.savefig('Fig4-plotmaxmin.pdf')

        # funcoes.save_hodograms(BA1,        BT1,        BR1,        BA2,  BT2,  BR2,  BA3,        BT3,        BR3)
        funcoes.save_hodograms(pltBy_id[0], pltBz_id[0], pltBx_id[0], pltBM, pltBL, pltBN, pltBy_id[1], pltBz_id[1],
                               pltBx_id[1],
                               pltBx_real, pltBy_real, pltBz_real,
                               vec_theta, vec_phi, theta_mva, phi_mva, table,
                               tipo, hand, eigval, eigvec, Pmetric, chi)

        #return [FIGURA4, '']

    if opt_plot == 'MVAinfo':
        plt.figure(numfig, dpi=300, figsize=(14, 12))
        FIGURA5 = figuras.figura5(tipo, hand, eigval, eigvec, phi_mva, theta_mva, vec_theta, vec_phi, Pmetric, chi, '')
        ##    plt.savefig('Fig4-'+NOMEARQ+'-'+tipo+'-'+hand+'H_juntos.pdf')
        # plt.close('all')
        # plt.savefig('Fig5-saidaMVA.pdf')
       # return [FIGURA5, '']

    if opt_plot == 'Helix':
        plt.figure(numfig, dpi=300, figsize=(14, 12))
        FIGURA6 = funcoes.funcao4(theta_mva, phi_mva, hand)
        ##    plt.savefig('Fig4-'+NOMEARQ+'-'+tipo+'-'+hand+'H_juntos.pdf')
        # plt.close('all')
        # plt.savefig('Fig6-Helix.pdf')
        #return [FIGURA6, '']




def plot_decision(opt_plot, sinalBA, sinalBT, decision):
    # abertura do arquivo
    arq = open('hodograms.txt', 'r')

    # Extrai dados linha a linha do arquivo arqdados
    dados = []
    count = 0
    arqcol = []
    vec_theta = np.zeros(2)
    vec_phi = np.zeros(2)
    eigval = np.zeros(3)
    eigvec = np.zeros([3, 3])

    for line in arq:

        if count == 1:  # vec_theta
            count = 0
            line2 = line[:-1].split(' ')
            vec_theta[0] = float(line2[0])
            vec_theta[1] = float(line2[1])

        if count == 2:  # vec_phi
            count = 0
            line2 = line[:-1].split(' ')
            vec_phi[0] = float(line2[0])
            vec_phi[1] = float(line2[1])

        if count == 3:  # vec_theta
            count = 0
            theta_mva = float(line[:-1])

        if count == 4:
            count = 0
            phi_mva = float(line[:-1])

        if count == 5:
            line2 = line[:-1].split(' ')
            while '' in line2:
                del line2[line2.index('')]  # filtro de espaços

            dados.append(line2)  # Copia dados em kip separando colunas

        if count == 6:
            count = 0
            Type = str(line[:-1])

        if count == 7:
            count = 0
            Handedness = str(line[:-1])

        if count == 8:
            count = 0
            line2 = line[:-1].split(' ')
            eigval[0] = float(line2[0])
            eigval[1] = float(line2[1])
            eigval[2] = float(line2[2])

        if count == 9:
            count = 0
            line2 = line[:-1].split(' ')
            eigvec[0, 0] = float(line2[0])
            eigvec[1, 0] = float(line2[1])
            eigvec[2, 0] = float(line2[2])

        if count == 10:
            count = 0
            line2 = line[:-1].split(' ')
            eigvec[0, 1] = sinalBA * float(line2[0])
            eigvec[1, 1] = sinalBA * float(line2[1])
            eigvec[2, 1] = sinalBA * float(line2[2])

        if count == 11:
            count = 0
            line2 = line[:-1].split(' ')
            eigvec[0, 2] = sinalBT * float(line2[0])
            eigvec[1, 2] = sinalBT * float(line2[1])
            eigvec[2, 2] = sinalBT * float(line2[2])

        if count == 12:
            count = 0
            Pmetric = float(line[:-1])

        if count == 13:
            count = 0
            chi = float(line[:-1])

        if line == 'vec_theta\n':
            count = 1
        elif line == 'vec_phi\n':
            count = 2
        elif line == 'theta_mva\n':
            count = 3
        elif line == 'phi_mva\n':
            count = 4
        elif line == 'BEGIN DATA\n':
            count = 5
        elif line == 'MCType\n':
            count = 6
        elif line == 'hand\n':
            count = 7
        elif line == 'eigval\n':
            count = 8
        elif line == 'eigvec1\n':
            count = 9
        elif line == 'eigvec2\n':
            count = 10
        elif line == 'eigvec3\n':
            count = 11
        elif line == 'Pmetric\n':
            count = 12
        elif line == 'chi\n':
            count = 13

    arq.close()

    ### Carrega dados nas variáveis desejadas

    # Inicializa vetores
    pltBN = np.zeros(len(dados))  # componente x do campo MVA
    pltBM = np.zeros(len(dados))  # componente y x do campo MVA
    pltBL = np.zeros(len(dados))  # componente z x do campo MVA
    pltBX0 = np.zeros(len(dados))  # componente x do campo IDEAL cenário 1
    pltBY0 = np.zeros(len(dados))  # componente y x do campo IDEAL cenário 1
    pltBZ0 = np.zeros(len(dados))  # componente z x do campo IDEAL cenário 1
    pltBX1 = np.zeros(len(dados))  # componente x do campo IDEAL cenário 2
    pltBY1 = np.zeros(len(dados))  # componente y x do campo IDEAL cenário 2
    pltBZ1 = np.zeros(len(dados))  # componente z x do campo IDEAL cenário 2
    pltBx_real = np.zeros(len(dados))  # componente z x do campo Ace Data
    pltBy_real = np.zeros(len(dados))  # componente z x do campo Ace Data
    pltBz_real = np.zeros(len(dados))  # componente z x do campo Ace Data

    # Carregamento linha a linha do arquivo DADOS
    cont = -1
    for x in range(len(dados)):
        cont = cont + 1
        # pltBy_id[0],pltBz_id[0],pltBx_id[0],pltBM,pltBL,pltBN,pltBy_id[1],pltBz_id[1],pltBx_id[1])
        # 0           1           2          3      4     5     6           7           8
        pltBY0[cont] = float(dados[x][0])
        pltBZ0[cont] = float(dados[x][1])
        pltBX0[cont] = float(dados[x][2])
        pltBM[cont] = float(dados[x][3])
        pltBL[cont] = float(dados[x][4])
        pltBN[cont] = float(dados[x][5])
        pltBY1[cont] = float(dados[x][6])
        pltBZ1[cont] = float(dados[x][7])
        pltBX1[cont] = float(dados[x][8])
        pltBx_real[cont] = float(dados[x][9])
        pltBy_real[cont] = float(dados[x][10])
        pltBz_real[cont] = float(dados[x][11])

    ### Verifica opção de PLOT

    numfig = 1
    if opt_plot == 'plotar_menosBA':
        eigvec[:, 1] = -eigvec[:, 1]
        phi = funcoes.calc_longitudinal(eigvec)
        phi_mva = phi[1]  # axial
        theta = funcoes.calc_latitudinal(eigvec)
        theta_mva = theta[1]  # axial

        plt.figure(numfig, dpi=300, figsize=(14, 12))  # troquei Bx, By Bz por BN BM BL
        FIGURA4 = figuras.figura4(pltBN, -pltBM, pltBL,
                                  pltBX0, pltBY0, pltBZ0,
                                  pltBX1, pltBY1, pltBZ1,
                                  vec_theta, vec_phi, theta_mva, phi_mva, ' $[-B_A,+B_T]$ ')
        ##    plt.savefig('Fig4-'+NOMEARQ+'-'+tipo+'-'+hand+'H_juntos.pdf')
        # plt.close('all')
        # plt.savefig('Fig4-plotmaxmin.pdf')

        # funcoes.save_hodograms(BA1,        BT1,        BR1,        BA2,  BT2,  BR2,  BA3,        BT3,        BR3)
        # funcoes.save_hodograms(pltBy_id[0],pltBz_id[0],pltBx_id[0],pltBM,pltBL,pltBN,pltBy_id[1],pltBz_id[1],pltBx_id[1])

        return [FIGURA4, '']

    if opt_plot == 'plotar_menosBT':
        eigvec[:, 2] = -eigvec[:, 2]
        phi = funcoes.calc_longitudinal(eigvec)
        phi_mva = phi[1]  # axial
        theta = funcoes.calc_latitudinal(eigvec)
        theta_mva = theta[1]  # axial

        plt.figure(numfig, dpi=300, figsize=(14, 12))  # troquei Bx, By Bz por BN BM BL
        FIGURA4 = figuras.figura4(pltBN, pltBM, -pltBL,
                                  pltBX0, pltBY0, pltBZ0,
                                  pltBX1, pltBY1, pltBZ1,
                                  vec_theta, vec_phi, theta_mva, phi_mva, ' $[+B_A,-B_T]$ ')
        ##    plt.savefig('Fig4-'+NOMEARQ+'-'+tipo+'-'+hand+'H_juntos.pdf')
        # plt.close('all')
        # plt.savefig('Fig4-plotmaxmin.pdf')

        # funcoes.save_hodograms(BA1,        BT1,        BR1,        BA2,  BT2,  BR2,  BA3,        BT3,        BR3)
        # funcoes.save_hodograms(pltBy_id[0],pltBz_id[0],pltBx_id[0],pltBM,pltBL,pltBN,pltBy_id[1],pltBz_id[1],pltBx_id[1])

        return [FIGURA4, '']

    if opt_plot == 'plotar_menosBAxBT':
        eigvec[:, 1] = -eigvec[:, 1]
        eigvec[:, 2] = -eigvec[:, 2]
        phi = funcoes.calc_longitudinal(eigvec)
        phi_mva = phi[1]  # axial
        theta = funcoes.calc_latitudinal(eigvec)
        theta_mva = theta[1]  # axial

        plt.figure(numfig, dpi=300, figsize=(14, 12))  # troquei Bx, By Bz por BN BM BL
        FIGURA4 = figuras.figura4(pltBN, -pltBM, -pltBL,
                                  pltBX0, pltBY0, pltBZ0,
                                  pltBX1, pltBY1, pltBZ1,
                                  vec_theta, vec_phi, theta_mva, phi_mva, ' $[-B_A,-B_T]$ ')
        ##    plt.savefig('Fig4-'+NOMEARQ+'-'+tipo+'-'+hand+'H_juntos.pdf')
        # plt.close('all')
        # plt.savefig('Fig4-plotmaxmin.pdf')

        # funcoes.save_hodograms(BA1,        BT1,        BR1,        BA2,  BT2,  BR2,  BA3,        BT3,        BR3)
        # funcoes.save_hodograms(pltBy_id[0],pltBz_id[0],pltBx_id[0],pltBM,pltBL,pltBN,pltBy_id[1],pltBz_id[1],pltBx_id[1])

        return [FIGURA4, '']

    if opt_plot == 'plotar_default':
        plt.figure(numfig, dpi=300, figsize=(14, 12))  # troquei Bx, By Bz por BN BM BL
        FIGURA4 = figuras.figura4(pltBN, pltBM, pltBL,
                                  pltBX0, pltBY0, pltBZ0,
                                  pltBX1, pltBY1, pltBZ1,
                                  vec_theta, vec_phi, theta_mva, phi_mva, ' default ')
        ##    plt.savefig('Fig4-'+NOMEARQ+'-'+tipo+'-'+hand+'H_juntos.pdf')
        # plt.close('all')
        # plt.savefig('Fig4-plotmaxmin.pdf')

        # funcoes.save_hodograms(BA1,        BT1,        BR1,        BA2,  BT2,  BR2,  BA3,        BT3,        BR3)
        # funcoes.save_hodograms(pltBy_id[0],pltBz_id[0],pltBx_id[0],pltBM,pltBL,pltBN,pltBy_id[1],pltBz_id[1],pltBx_id[1])

        return [FIGURA4, '']

    if opt_plot == 'plotar_mvaresults':
        phi = funcoes.calc_longitudinal(eigvec)
        phi_mva = phi[1]  # axial
        theta = funcoes.calc_latitudinal(eigvec)
        theta_mva = theta[1]  # axial

        if decision == 1:
            if sinalBA == -1:
                if sinalBT == -1:
                    decision_info = ' with decision: $[-B_A,-B_T]$'
                else:
                    decision_info = ' with decision: $[-B_A,+B_T]$'
            elif sinalBT == -1:
                decision_info = ' with decision: $[+B_A,-B_T]$'
            else:
                decision_info = ' with decision: $[+B_A,+B_T]$'
        else:
            decision_info = ''

        FIGURA5 = figuras.figura5(Type, Handedness, eigval, eigvec, phi_mva, theta_mva,
                                  vec_theta, vec_phi, Pmetric, chi, decision_info)
        return [FIGURA5, '']

    if opt_plot == 'plotar_axisdirections':
        phi = funcoes.calc_longitudinal(eigvec)
        phi_mva = phi[1]  # axial
        theta = funcoes.calc_latitudinal(eigvec)
        theta_mva = theta[1]  # axial

        if decision == 1:
            if sinalBA == -1:
                if sinalBT == -1:
                    decision_info = ' with decision: $[-B_A,-B_T]$'
                else:
                    decision_info = ' with decision: $[-B_A,+B_T]$'
            elif sinalBT == -1:
                decision_info = ' with decision: $[+B_A,-B_T]$'
            else:
                decision_info = ' with decision: $[+B_A,+B_T]$'
        else:
            decision_info = ''

        pltmodB_real = np.sqrt(pltBx_real ** 2 + pltBy_real ** 2 + pltBz_real ** 2)
        pltBlat_real, pltBlon_real = funcoes.funcao3(pltBx_real, pltBy_real, pltBz_real)

        pltBx_id0 = pltBX0
        pltBy_id0 = pltBY0  # sinalBA*pltBY0
        pltBz_id0 = pltBZ0  # sinalBT*pltBZ0

        pltmodB_id0 = np.sqrt(pltBx_id0 ** 2 + pltBy_id0 ** 2 + pltBz_id0 ** 2)
        pltBlat_id0, pltBlon_id0 = funcoes.funcao3(pltBx_id0, pltBy_id0, pltBz_id0)

        pltBx_id1 = pltBX1
        pltBy_id1 = pltBY1  # sinalBA*pltBY1
        pltBz_id1 = pltBZ1  # sinalBT*pltBZ1

        pltmodB_id1 = np.sqrt(pltBx_id1 ** 2 + pltBy_id1 ** 2 + pltBz_id1 ** 2)
        pltBlat_id1, pltBlon_id1 = funcoes.funcao3(pltBx_id1, pltBy_id1, pltBz_id1)

        FIGURA3 = figuras.figura3(pltmodB_real, pltBx_real, pltBy_real, pltBz_real, pltBlat_real, pltBlon_real,
                                  pltmodB_id0, pltBx_id0, pltBy_id0, pltBz_id0, pltBlat_id0, pltBlon_id0,
                                  pltmodB_id1, pltBx_id1, pltBy_id1, pltBz_id1, pltBlat_id1, pltBlon_id1,
                                  vec_theta, vec_phi, phi_mva, theta_mva, decision_info)
        return [FIGURA3, '']

