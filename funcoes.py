# Este programa tem por finalidade definir as funções
# que serão usadas nos programas
# Programa 1 - para forcefree + rotação -> opções de Max/Min
# Programa 2 - para ACEdata + MVA + MCtype + forcefree + rotação -> Max/Min real e ideal

import numpy as np
import os
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
import mpmath as mp
import math  # para fatorial, pi, cos, sin
import pytest
from mpl_toolkits.mplot3d import axes3d
import geometry
import datetime


def save_hodograms(BA1, BT1, BR1, BA2, BT2, BR2, BA3, BT3, BR3, Bx, By, Bz, vec_theta, vec_phi, theta_mva, phi_mva,
                   inicio_fim,
                   tipo, hand, eigval, eigvec, Pmetric, chi):
    # inicio_fim = [data_inicio[2],data_inicio[1],data_inicio[0],str(hora_inicio),data_fim[2],data_fim[1],data_fim[0],str(hora_fim)]

    arq = open("hodograms.txt", "w")


    # arq.write(Decision)
    arq.write("\n")
    arq.write("MCType\n")
    arq.write(str(tipo) + '\n')
    arq.write("hand\n")
    arq.write(str(hand) + '\n')
    arq.write("eigval\n")
    arq.write(str(eigval[0]) + ' ' + str(eigval[1]) + ' ' + str(eigval[2]) + '\n')
    arq.write("eigvec1\n")
    arq.write(str(eigvec[0, 0]) + ' ' + str(eigvec[1, 0]) + ' ' + str(
        eigvec[2, 0]) + ' ' + '\n')  # os autovetores estão nas colunas
    arq.write("eigvec2\n")
    arq.write(str(eigvec[0, 1]) + ' ' + str(eigvec[1, 1]) + ' ' + str(
        eigvec[2, 1]) + ' ' + '\n')  # os autovetores estão nas colunas
    arq.write("eigvec3\n")
    arq.write(str(eigvec[0, 2]) + ' ' + str(eigvec[1, 2]) + ' ' + str(
        eigvec[2, 2]) + ' ' + '\n')  # os autovetores estão nas colunas
    arq.write("Pmetric\n")
    arq.write(str(Pmetric) + '\n')
    arq.write("chi\n")
    arq.write(str(chi) + '\n')
    arq.write("ano_inicio\n")
    arq.write(str(inicio_fim[0]) + '\n')
    arq.write("mes_inicio\n")
    arq.write(str(inicio_fim[1]) + '\n')
    arq.write("dia_inicio\n")
    arq.write(str(inicio_fim[2]) + '\n')
    arq.write("hora_inicio\n")
    arq.write(str(inicio_fim[3]) + '\n')
    arq.write("ano_fim\n")
    arq.write(str(inicio_fim[4]) + '\n')
    arq.write("mes_fim\n")
    arq.write(str(inicio_fim[5]) + '\n')
    arq.write("dia_fim\n")
    arq.write(str(inicio_fim[6]) + '\n')
    arq.write("hora_fim\n")
    arq.write(str(inicio_fim[7]) + '\n')
    arq.write("vec_theta\n")
    arq.write(str(vec_theta[0]) + ' ' + str(vec_theta[1]) + '\n')
    arq.write("vec_phi\n")
    arq.write(str(vec_phi[0]) + ' ' + str(vec_phi[1]) + '\n')
    arq.write("theta_mva\n")
    arq.write(str(theta_mva) + '\n')
    arq.write("phi_mva\n")
    arq.write(str(phi_mva) + '\n')
    arq.write("BA1 BT1 BR1 BA2 BT2 BR2 BA3 BT3 BR3 Bx_AceData By_AceData Bz_AceData\n")
    arq.write("BEGIN DATA\n")

    # Escrita dos dados
    Linha = ''
    for x in range(len(BA1)):
        Linha = Linha + str(BA1[x]) + ' ' + str(BT1[x]) + ' ' + str(BR1[x]) + ' ' + str(BA2[x]) + ' ' + str(
            BT2[x]) + ' ' + str(BR2[x]) + ' ' + str(BA3[x]) + ' ' + str(BT3[x]) + ' ' + str(BR3[x]) + ' ' + str(
            Bx[x]) + ' ' + str(By[x]) + ' ' + str(Bz[x]) + '\n'
        arq.write(Linha)  # Escreve cada Linha no arquivo
        Linha = ''

    arq.close()

    return


def beta_plasma(Bx, By, Bz, Np, Tp):
    beta = np.zeros(len(Bx))
    mu0 = 4 * np.pi * 10 ** (-7)
    k = 1.38065 * 10 ** (-23)

    for i in range(len(Bx)):
        Pn = (Bx[i] * Bx[i] + By[i] * By[i] + Bz[i] * Bz[i]) * 10 ** (-18) / (2 * mu0)

        Pk = Np[i] * Tp[i] * k * 10 ** 6

        beta[i] = Pk / Pn

    return beta


def metric(eigval):
    #### artigo submetido em Ago/2019 ####
    ######################
    ##### processamento ##
    ##### DEGENERACAO ####
    ######################

    lamb1 = abs(eigval[0] / eigval[0])  # igual a 1

    lamb2 = abs(eigval[1] / eigval[0])

    lamb3 = abs(eigval[2] / eigval[0])

    # ACHATAMENTOS
    A = (1 - np.sqrt(lamb3))
    B = (1 - np.sqrt(lamb2))
    C = abs((np.sqrt(lamb2) - np.sqrt(lamb3)))
    Z = A * B * C * 100

    return Z


def anglelist(phi, theta, MCType):  # aqui: phi(LONG - 0 a 360)  theta(LAT - -90 a +90)
    #####################################
    # Obtém Lista de Ângulos para "processamento ideal"
    # resultando nas (2) opções de Max/Min
    # SAÍDA: vec_theta, vec_phi, vec_hand
    #####################################

    if MCType == 'NES':
        vec_hand = ['R', 'R']
        if phi > 180.:
            phi = phi - 180.
        vec_phi = [math.radians(phi), math.radians(phi)]
        vec_theta = [math.radians(theta), math.radians(-theta)]

    elif MCType == 'NWS':
        vec_hand = ['L', 'L']
        if phi < 180:
            phi = phi + 180.
        vec_phi = [math.radians(phi), math.radians(phi)]
        vec_theta = [math.radians(theta), math.radians(-theta)]

    elif MCType == 'SEN':
        vec_hand = ['L', 'L']
        if phi > 180:
            phi = phi - 180.
        vec_phi = [math.radians(phi), math.radians(phi)]
        vec_theta = [math.radians(theta), math.radians(-theta)]

    elif MCType == 'SWN':
        vec_hand = ['R', 'R']
        if phi < 180:
            phi = phi + 180.
        vec_phi = [math.radians(phi), math.radians(phi)]
        vec_theta = [math.radians(theta), math.radians(-theta)]

    elif MCType == 'ENW':
        vec_hand = ['L', 'L']
        if theta < 0:
            theta = -theta
        vec_theta = [math.radians(theta), math.radians(theta)]
        if phi < 180:
            vec_phi = [math.radians(phi), math.radians(phi + 180)]
        else:
            vec_phi = [math.radians(phi), math.radians(phi - 180)]
    elif MCType == 'ESW':
        vec_hand = ['R', 'R']
        if theta > 0:
            theta = -theta
        vec_theta = [math.radians(theta), math.radians(theta)]
        if phi < 180:
            vec_phi = [math.radians(phi), math.radians(phi + 180)]
        else:
            vec_phi = [math.radians(phi), math.radians(phi - 180)]

    elif MCType == 'WNE':
        vec_hand = ['R', 'R']
        if theta < 0:
            theta = -theta
        vec_theta = [math.radians(theta), math.radians(theta)]
        if phi < 180:
            vec_phi = [math.radians(phi), math.radians(phi + 180)]
        else:
            vec_phi = [math.radians(phi), math.radians(phi - 180)]

    elif MCType == 'WSE':
        vec_hand = ['L', 'L']
        if theta > 0:
            theta = -theta
        vec_theta = [math.radians(theta), math.radians(theta)]
        if phi < 180:
            vec_phi = [math.radians(phi), math.radians(phi + 180)]
        else:
            vec_phi = [math.radians(phi), math.radians(phi - 180)]

    return vec_phi, vec_theta, vec_hand


def funcaotype(Lat, Lon, theta):
    #####################################
    # Identificação automática do tipo de tubo
    # Aplicado no caso ideal (diretamente) e no caso real (após filtro)
    # SAÍDA: tipo, hand
    #####################################

    limear = 100  # Teste descontinuidade
    posdesc = 0  # Posicao de descontinuidade
    crescLat = 0  # votacao crescimento
    decrescLat = 0  # votacao decrescimento
    crescLon = 0  # votacao crescimento
    decrescLon = 0  # votacao decrescimento

    for x in range(len(Lon) - 1):
        dif1 = Lon[x + 1] - Lon[x]
        dif2 = Lat[x + 1] - Lat[x]
        if dif1 > limear:
            posdesc = x  # posicao de descontinuidade
            crescLon = crescLon + 1  # votacao crescente
        if dif1 < -limear:
            posdesc = x  # posicao de descontinuidade
            decrescLon = decrescLon + 1  # votacao decrescente
        if dif2 > 0:
            crescLat = crescLat + 1  # votacao crescente
        if dif2 < 0:
            decrescLat = decrescLat + 1  # votacao decrescente

    if abs(theta) < np.pi / 4 or abs(theta) == np.pi / 4:
        # posdesc == 0: não usamos mais a descontinuidade, apenas o theta de acordo com Arian.
        # Continuo (TUBO BIPOLAR)
        if crescLat > decrescLat:
            # Latitude crescente
            if np.mean(Lon) > 180:
                tipo = 'SWN'
                hand = 'R'
            else:
                tipo = 'SEN'
                hand = 'L'
        else:
            # Latitude decrescente
            if np.mean(Lon) > 180:
                tipo = 'NWS'
                hand = 'L'
            else:
                tipo = 'NES'
                hand = 'R'
    else:
        # Descontinuo (TUBO UNIPOLAR)
        if crescLon > decrescLon:
            # Longitude crescente
            # if np.mean(Lat[posdesc-5:posdesc+5]) > 0:
            if np.mean(Lat) > 0:
                tipo = 'ENW'
                hand = 'L'
            else:
                tipo = 'ESW'
                hand = 'R'
        else:
            # Longitude decrescente
            # if np.mean(Lat[posdesc-5:posdesc+5]) > 0:
            if np.mean(Lat) > 0:
                tipo = 'WNE'
                hand = 'R'
            else:
                tipo = 'WSE'
                hand = 'L'

    return tipo, hand


def funcao0(phi, theta):  # aqui: phi(LONG - 0 a 360)  theta(LAT - -90 a +90)
    #####################################
    # Definição do ângulo omega (de projeção no plano yOz)
    # Pré-requisito para rotação 3D (funcao1) no caso ideal
    # SAÍDA: omega
    #####################################

    if 0 < theta < np.pi / 2:
        if 0 < phi < np.pi:
            omega = theta
        if np.pi < phi < 2 * np.pi:
            omega = np.pi - theta
        if phi == 0 or phi == np.pi or phi == 2 * np.pi:
            omega = np.pi / 2
    if -np.pi / 2 < theta < 0:
        if 0 < phi < np.pi:
            omega = 2 * np.pi + theta
        if np.pi < phi < 2 * np.pi:
            omega = np.pi - theta
        if phi == 0 or phi == np.pi or phi == 2 * np.pi:
            omega = 3 * np.pi / 2
    if theta == 0:
        if 0 < phi < np.pi:
            omega = 0
        if np.pi < phi < 2 * np.pi:
            omega = np.pi
        if phi == 0 or phi == np.pi or phi == 2 * np.pi:
            print('Não existe omega')
    if theta == -np.pi / 2:
        omega = 3 * np.pi / 2
    if theta == np.pi / 2:
        omega = np.pi / 2

    return omega


def funcao1(B0, alpha, H, r_max, omega):
    #####################################
    # Modelo Force-free e Rotação 3D dos dados
    # Pré-requisito para "processamento ideal"
    # SAÍDA: r, BA, BT, BR, modB, ROT, Bx, By, Bz
    #####################################

    e1 = np.array([1, 0., 0.])
    e2 = np.array([0., 1., 0.])
    e3 = np.array([0., 0., 1.])

    # Prepara matriz de rotação 3D ROT usando theta e phi
    '''1º rotaciona vetor de theta em torno z (vetor*) e y de theta em torno de z (y*)'''
    M1 = geometry.rotation_matrix_from_axis_and_angle(e1, omega)  # essa rotacao é um giro
    e1 = np.dot(M1, e1)
    e2 = np.dot(M1, e2)
    e3 = np.dot(M1, e3)

    ROT = np.transpose(np.array([e1, e2, e3]))
    ##
    # Cria os coeficientes de Bessel como modelo Force-Free para campo magnetico de nuvens magneticas

    r_min = -r_max
    r_qtd = (r_max - r_min) * 10 + 1  # Define passo 0.1
    r = np.linspace(r_min, r_max, r_qtd)  # discretiza intervalo r_min a r_max com r_qtd valores

    ## Construção das funções de Bessel
    # BesselJ(K,r,LIM) -> K=ordem, x=variavel, LIM= limitante da série "infinita"
    LIM = 2 * r_max

    # BA = B0*BesselJ[0,alpha*r]
    # BT = B0*BesselJ[1,H*alpha*r]
    # BR = 0

    BA = np.zeros(r_qtd)
    By = np.zeros(r_qtd)
    xA = alpha * r

    BT = np.zeros(r_qtd)
    Bz = np.zeros(r_qtd)
    xT = H * alpha * r

    BR = np.zeros(r_qtd)
    Bx = np.zeros(r_qtd)

    for i in range(r_qtd):
        for m in range(LIM):
            BA[i] = (BA[i] + (((-1) ** (m)) * (xA[i] ** (2 * m + 0)) /
                              (2 ** (2 * m + 0) * math.factorial(m) * math.factorial(m + 0))))
            BT[i] = (BT[i] + (((-1) ** (m)) * (xT[i] ** (2 * m + 1)) /
                              (2 ** (2 * m + 1) * math.factorial(m) * math.factorial(m + 1))))

    for i in range(r_qtd):
        ## Rotacionar BA, BT e BR
        [BR[i], BA[i], BT[i]] = np.dot(ROT, [BR[i], BA[i], BT[i]]) + 0.001

    #    print('ROT', ROT)
    BR = B0 * BR
    BA = B0 * BA
    BT = B0 * BT

    Bx = BR
    By = BA
    Bz = BT

    ## |B|=SQRT[BA^2+BT^2]
    modB = (BA ** 2 + BT ** 2 + BR ** 2) ** (1 / 2)

    return r, BA, BT, BR, modB, ROT, Bx, By, Bz


def funcao2(Bx, By, Bz):
    #####################################
    # MVA
    # Aplicado aos dados reais no intervalo da nuvem
    # Pré-requisito para "processamento real"
    # SAÍDA: BL, BM, BN, eigval, eigvec, phiaxi, thetaaxi
    #####################################

    # Abre arquivo de Print de informações
    # arq=open("SaidaMVA.txt","w")

    # Inicializa Matriz
    B1 = np.array([Bx, By, Bz])  # Bl Bx, By e Bz nas linhas

    B1 = np.transpose(B1)  # agora, Bl com Bx, By e Bz nas colunas

    (m, n) = np.shape(B1)

    mu = np.mean(B1, 0)  # mu=(<Bx>,<By>,<Bz>)

    # Inicializa Matriz
    B = np.zeros([m, n])

    # tirar a media sem usar o for
    for i in range(m):
        for j in range(n):
            B[i, j] = B1[i, j] - mu[j]  # B=B1-mu

    Q = np.dot(np.transpose(B), B) / m  # Q = BT*B/N

    # Obter autovalores e autovetores
    [eigval, eigvec] = np.linalg.eig(Q)

    # Ordenar autovalores sem perder correspondencia com autovetores.
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    r = eigval[1] / (eigval[2] + 0.001)

    # print('Razao lamb2/lamb3='+str(r))
    # arq.write('Razao lamb2/lamb3='+str(r)+'\n')

    phi = calc_longitudinal(eigvec)

    # Ângulo Longitudinal Phi
    #    phi = np.zeros(3)

    #    for i in range(3):
    #        Bx = eigvec[0][i]
    #        By = eigvec[1][i]
    #        if Bx > 0 and (By > 0 or By == 0):
    #            phi[i] = (mp.atan(By/Bx)*180)/np.pi
    #        if Bx < 0 and (By > 0):
    #            phi[i] = 180 - mp.atan(By/abs(Bx))*180/np.pi
    #        if Bx < 0 and (By < 0 or By == 0):
    #            phi[i] = 180 + mp.atan(abs(By)/abs(Bx))*180/np.pi
    #        if Bx > 0 and (By < 0):
    #            phi[i] = 360 - mp.atan(abs(By)/Bx)*180/np.pi
    #        if Bx == 0 and (By > 0):
    #            phi[i] = 90
    #        #if Bx < 0 and (By == 0):
    #        #   phi[i] = 180
    #        if Bx == 0 and (By < 0):
    #            phi[i] = 270
    #        if Bx == 0 and (By == 0):
    #            phi[i] = 0

    phimax = phi[0]
    phiaxi = phi[1]
    phimin = phi[2]

    theta = calc_latitudinal(eigvec)

    ##    #Ângulo Latitudinal theta
    ##    theta = np.zeros(3)
    ##    for i in range(3):
    ##        Bx = eigvec[0][i]
    ##        By = eigvec[1][i]
    ##        Bz = eigvec[2][i]
    ##        if Bx == 0 and By == 0:
    ##            if Bz > 0:
    ##                theta[i] = 90
    ##            if Bz < 0:
    ##                theta[i] = -90
    ##        else:
    ##            angle = mp.atan(abs(Bz)/((Bx**2+By**2)**(1/2)))*180/np.pi
    ##            if (Bz > 0 or Bz == 0):
    ##                theta[i] = angle
    ##            else:
    ##                theta[i] = -angle #estamos interpretando theta do plano xOy para os polos z+/z-

    thetamax = theta[0]
    thetaaxi = theta[1]
    thetamin = theta[2]

    # print('Ângulo phi máximo: '+ str(phimax))
    # arq.write('Ângulo phi máximo: '+ str(phimax)+'\n')
    # print('Ângulo phi axial: '+ str(phiaxi))
    # arq.write('Ângulo phi axial: '+ str(phiaxi)+'\n')
    # print('Ângulo phi mínimo: '+ str(phimin))
    # arq.write('Ângulo phi mínimo: '+ str(phimin)+'\n')

    # print('Ângulo theta máximo: '+ str(thetamax))
    # arq.write('Ângulo theta máximo: '+ str(thetamax)+'\n')
    # print('Ângulo theta axial: '+ str(thetaaxi))
    # arq.write('Ângulo theta axial: '+ str(thetaaxi)+'\n')
    # print('Ângulo theta mínimo: '+ str(thetamin))
    # arq.write('Ângulo theta mínimo: '+ str(thetamin)+'\n')

    #    print('Autovalores: '+ str(eigval))
    # arq.write('Autovalores: '+ str(eigval)+'\n')
    #    print('Autovetor Max: '+ str(eigvec[:,0])) #os autovetores estão nas colunas
    # arq.write('Autovetor Max: '+ str(eigvec[:,0])+'\n')
    #    print('Autovetor Interm: '+ str(eigvec[:,1])) #os autovetores estão nas colunas
    # arq.write('Autovetor Interm: '+ str(eigvec[:,1])+'\n')
    #    print('Autovetor Min: '+ str(eigvec[:,2])) #os autovetores estão nas colunas
    # arq.write('Autovetor Min: '+ str(eigvec[:,2])+'\n')
    #
    Y = np.dot(B, eigvec)  # Y=[Bx*,By*,Bz*]=[BR,BA,BT]=[BL,BM,BN] nas colunas
    BL = Y[:, 0]
    BM = Y[:, 1]
    BN = Y[:, 2]
    ##### para visualização correta dos planos de máx e mín a correspondência dessa saída é
    #### Bx* -> BN, By* -> BM, Bz* -> BL.
    #### o plano max é BL X BM (ou Bz* X By*)
    #### o plano min é BM X BN (ou By* X Bx*)

    # Cálculo do ângulo chi

    # chi = mp.acos(np.dot(BL, BN)/(np.linalg.norm(BL)* np.linalg.norm(BN)))*180/np.pi

    chi = mp.acos(np.dot(B1[0, :], B1[-1, :]) / (np.linalg.norm(B1[0, :]) * np.linalg.norm(B1[-1, :]))) * 180 / np.pi

    #    print('Angulo chi: '+ str(chi))
    #    arq.write('Angulo chi: '+ str(chi)+'\n')

    # if (r>2 or r==2) and (chi>30 or chi ==30):
    # print('O método da mínima variância é consistente')
    # arq.write('O método da mínima variância é consistente\n')
    # else:
    # print('O método da mínima variância não é consistente')
    # arq.write("O método da mínima variância não é consistente\n")

    #    arq.close()

    return BL, BM, BN, eigval, eigvec, phiaxi, thetaaxi, chi


def calc_longitudinal(eigvec):
    # Ângulo Longitudinal Phi
    phi = np.zeros(3)

    for i in range(3):
        Bx = eigvec[0][i]
        By = eigvec[1][i]
        if Bx > 0 and (By > 0 or By == 0):
            phi[i] = (mp.atan(By / Bx) * 180) / np.pi
        if Bx < 0 and (By > 0):
            phi[i] = 180 - mp.atan(By / abs(Bx)) * 180 / np.pi
        if Bx < 0 and (By < 0 or By == 0):
            phi[i] = 180 + mp.atan(abs(By) / abs(Bx)) * 180 / np.pi
        if Bx > 0 and (By < 0):
            phi[i] = 360 - mp.atan(abs(By) / Bx) * 180 / np.pi
        if Bx == 0 and (By > 0):
            phi[i] = 90
        # if Bx < 0 and (By == 0):
        #   phi[i] = 180
        if Bx == 0 and (By < 0):
            phi[i] = 270
        if Bx == 0 and (By == 0):
            phi[i] = 0

    return phi


def calc_latitudinal(eigvec):
    # Ângulo Latitudinal theta
    theta = np.zeros(3)
    for i in range(3):
        Bx = eigvec[0][i]
        By = eigvec[1][i]
        Bz = eigvec[2][i]
        if Bx == 0 and By == 0:
            if Bz > 0:
                theta[i] = 90
            if Bz < 0:
                theta[i] = -90
        else:
            angle = mp.atan(abs(Bz) / ((Bx ** 2 + By ** 2) ** (1 / 2))) * 180 / np.pi
            if (Bz > 0 or Bz == 0):
                theta[i] = angle
            else:
                theta[i] = -angle  # estamos interpretando theta do plano xOy para os polos z+/z-
    return theta


def funcao3(Bx, By, Bz):
    #####################################
    # Cálculo das componentes Blat e Blong do campo B
    # Aplicado ao caso ideal (diretamente) e ao caso real (diretamente)
    # SAÍDA: Blat, Blon
    #####################################

    n = len(Bx)
    Blon = np.zeros(n)
    Blat = np.zeros(n)

    for x in range(n):
        # Cálculo de Blat
        Blat[x] = mp.acot((Bx[x] ** 2 + By[x] ** 2) ** (1 / 2) / Bz[x]) * 180 / np.pi

        # Cálculo de Blong - -90 < Blong < 90, mas precisamos de 0 < Blong < 360)
        if Bx[x] > 0 and (By[x] > 0 or By[x] == 0):
            Blon[x] = (mp.atan(By[x] / Bx[x]) * 180) / np.pi
        if Bx[x] < 0 and (By[x] > 0):
            Blon[x] = 180 - mp.atan(By[x] / abs(Bx[x])) * 180 / np.pi
        if Bx[x] < 0 and (By[x] < 0):
            Blon[x] = 180 + mp.atan(abs(By[x]) / abs(Bx[x])) * 180 / np.pi
        if Bx[x] > 0 and (By[x] < 0):
            Blon[x] = 360 - mp.atan(abs(By[x]) / Bx[x]) * 180 / np.pi
        if Bx[x] == 0 and (By[x] > 0):
            Blon[x] = 90
        if Bx[x] < 0 and (By[x] == 0):
            Blon[x] = 180
        if Bx[x] == 0 and (By[x] < 0):
            Blon[x] = 270
        if Bx[x] == 0 and (By[x] == 0):
            Blon[x] = 0

    Blat[0] = Blat[1]  # = Blat[2]
    Blat[-1] = Blat[-2]  # = Blat[-3]

    Blon[0] = Blon[1] = Blon[2]
    Blon[-1] = Blon[-2]  # = Blon[-3]

    return Blat, Blon


def funcao4(theta, phi, hand):
    #####################################
    # Obtém Cilindro (hélice) correspondente ao modelo ideal
    # Aplicado ao caso ideal (diretamente) e ao caso real (diretamente)
    # Define sistema de eixos (do espaço que contém a hélice) e calcula a hélice (a partir da equação vetorial)
    # Variável Xs guarda as coordenadas do cilindro (hélice)
    # SAÍDA: fig (figura pronta para visualizar ou salvar)
    #####################################

    # parâmetros
    N = 100
    k3 = 16
    C = np.array([0, 0, 0])
    r = 1

    '''1º rotaciona vetor de theta em torno z (vetor*) e y de theta em torno de z (y*)'''
    e1 = np.array([1., 0., 0.])
    e2 = np.array([0., 1., 0.])
    e3 = np.array([0., 0., 1.])

    # Preparando Sistemas de Eixos
    z = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
    x = np.cross(z, e1)
    y = np.cross(z, x)

    '''"tamanho" (em radianos) do passo de discretização da hélice'''
    delta = np.radians(10)

    # Processamento de fato
    '''"tamanho" do passo de discretização da hélice no eixo z'''
    t = 0.05
    if (hand == 'l' or hand == 'L' or hand == 'left' or hand == 'Left' or hand == 'LEFT'):
        # left hand
        Xs = np.array(
            [C + r * (i * 5 * t * z + 5 * np.sin(i * delta * 2) * x + 5 * np.cos(i * delta * 2) * y) for i in range(N)])

    if (hand == 'r' or hand == 'R' or hand == 'right' or hand == 'Right' or hand == 'RIGHT'):
        # right hand
        Xs = np.array(
            [C + r * (i * 5 * t * z + 5 * np.cos(i * delta * 2) * x + 5 * np.sin(i * delta * 2) * y) for i in range(N)])

    # eixo do cilindro
    w = z

    # Plotando
    fig = plt.figure(1, figsize=(17, 8))
    ax = fig.gca(projection='3d')
    ax.plot(Xs[:, 0], Xs[:, 1], Xs[:, 2], c='gray', linewidth=2)
    # Plot the center and direction
    k1 = 10
    k2 = 2
    ax.quiver(Xs[:-k2:k1, 0], Xs[:-k2:k1, 1], Xs[:-k2:k1, 2],
              Xs[k2::k1, 0] - Xs[:-k2:k1, 0], Xs[k2::k1, 1] - Xs[:-k2:k1, 1], Xs[k2::k1, 2] - Xs[:-k2:k1, 2],
              color='r', linewidth=3)

    ax.quiver(C[0], C[1], C[2],
              w[0] * N * 0.25, w[1] * N * 0.25, w[2] * N * 0.25, color='k')

    plt.plot([10, -10, 0], [0, 0, 0], [0, 0, 0], color='blue', linewidth=2, linestyle='-')
    plt.plot([0, 0, 0], [10, -10, 0], [0, 0, 0], color='blue', linewidth=2, linestyle='-')
    plt.plot([0, 0, 0], [0, 0, 0], [10, -10, 0], color='blue', linewidth=2, linestyle='-')

    ax.text(e1[0] * N * 0.15, e1[1] * N * 0.15, e1[2] * N * 0.15, 'SUN', size=10, zorder=1, color='b')  # label para SOL
    ax.text(-e1[0] * N * 0.15, -e1[1] * N * 0.15, -e1[2] * N * 0.15, 'EARTH', size=10, zorder=1,
            color='b')  # label para TERRA

    ax.text(e2[0] * N * 0.15, e2[1] * N * 0.15, e2[2] * N * 0.15, 'EAST', size=10, zorder=1,
            color='b')  # label para EAST
    ax.text(-e2[0] * N * 0.15, -e2[1] * N * 0.15, -e2[2] * N * 0.15, 'WEST', size=10, zorder=1,
            color='b')  # label para WEST

    ax.text(e3[0] * N * 0.15, e3[1] * N * 0.15, e3[2] * N * 0.15, 'NORTH', size=10, zorder=1,
            color='b')  # label para NORTH
    ax.text(-e3[0] * N * 0.15, -e3[1] * N * 0.15, -e3[2] * N * 0.15, 'SOUTH', size=10, zorder=1,
            color='b')  # label para SOUTH

    # Plot dos eixos coordenados
    ax.set_xlim([-k3, k3])
    ax.set_ylim([-k3, k3])
    ax.set_zlim([-k3, k3])

    ax.tick_params(labelbottom=False, labelleft=False)
    ax.grid(False)

    plt.suptitle(r'$\phi:$' + str(int(theta * 180 / np.pi)) + '°, ' + r'$\theta:$' + str(
        int(phi * 180 / np.pi)) + '°, Hand: ' + hand, fontsize=14)
    #    plt.title('theta= '+str(theta*180/np.pi)+'º and phi= '+str(phi*180/np.pi)+'° and hand= '+ hand+'.')
    return fig


# COMO INSERIR TEXTOS MATPLOTLIB3D: https://matplotlib.org/examples/mplot3d/text3d_demo.html


def is_leap_year(year):
    # Determina se é ano bissexto
    # Saída: 0/1 (valor lógico s/n)
    """ if year is a leap year return True
        else return False """
    if year % 100 == 0:
        return year % 400 == 0
    return year % 4 == 0


def doy(Y, M, D):
    # Conversor de dia/mes/ano para dia do ano.
    # SAÍDA: N (inteiro com doy)
    """ given year, month, day return day of year
        Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7 """
    if is_leap_year(Y):
        K = 1
    else:
        K = 2
    N = int((275 * M) / 9.0) - K * int((M + 9) / 12.0) + D - 30
    return N


def doytodate(Y, First, N):
    # Conversor de dia do ano para dia/mes/ano.
    """ given year, Nth day of year and return year, month and day"""
    D1 = datetime.date(Y, 1, 1) + datetime.timedelta(First - 1)
    D = datetime.date(D1.year, D1.month, D1.day) + datetime.timedelta(N)

    return D.year, D.month, D.day


def corte(cinicio, cfim, xinicio, xfim):
    # Calculadora para experimentos de determinação de início e fim da Nuvem
    """ dados corte inicio/fim e xinicio/xfim atuais
    retorna os novos valores para xinicio e xfim"""
    tam = xfim - xinicio
    return xinicio + cinicio, tam + cfim
