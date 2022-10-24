# Este programa tem por finalidade padronizar as figuras
# que serão usadas nos programas
# Programa 1 - para forcefree + rotação -> opções de Max/Min
# Programa 2 - para ACEdata + MVA + MCtype + forcefree + rotação -> Max/Min real e ideal

import numpy as np
import os
import matplotlib

matplotlib.use('TkAgg')
# matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
import mpmath as mp
import math  # para fatorial, pi, cos, sin
import pytest
from mpl_toolkits.mplot3d import axes3d
import geometry
import datetime
import funcoes


def figuraideal(BLat, BLon, Bx, By, Bz, phi, theta, hand, tipo):
    # Figura com os cenários ideiais
    # dos planos de máxima e de mínima variância,
    # componentes Blat e BLon do campo.

    fs = 14  # fontsize para gráficos

    numfig = 1
    fig = plt.figure(numfig, figsize=(18, 5))
    #    plt.suptitle(r'$\phi:$'+str(int(theta))+'°, '+r'$\theta:$'+str(int(phi))+'°, Hand: '+hand+', '+tipo,
    #              fontsize=fs)
    plt.suptitle(r'$\phi:$' + str(int(theta)) + '°, ' + r'$\theta:$' + str(int(phi)) + '°, Hand: ' + hand,
                 fontsize=fs)

    plt.subplots_adjust(hspace=0.4)  # Distância entre subplots
    sub1 = plt.subplot(2, 2, 1)
    if np.max(By) - np.min(By):
        amplitudey = np.max(By) - np.min(By)
    else:
        amplitudey = 1

    if np.max(Bz) - np.min(Bz):
        amplitudez = np.max(Bz) - np.min(Bz)
    else:
        amplitudez = 1

    normBy = (By - np.mean(By)) / amplitudey
    normBz = (Bz - np.mean(Bz)) / amplitudez
    # plt.plot(Bz/amplitudez, By/amplitudez,'-k', linewidth=2)
    plt.plot(normBz, normBy, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz) / n)
    for i in range(n):
        plt.plot(normBz[i * k], normBy[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    #    plt.plot(normBz[::k],normBy[::k],'ob')
    #    plt.plot(normBz[0],normBy[0],marker='o',
    #                 markersize=6,markerfacecolor='k',
    #                 markeredgewidth=2,markeredgecolor='k') #Begin
    plt.plot(normBz[-1], normBy[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    # plt.ylabel('$B_R(B^*_x)$',fontsize=fs)
    plt.xlabel('$B_T(B^*_z)$', fontsize=fs)
    plt.ylabel('$B_A(B^*_y)$', fontsize=fs)
    # Linha zero
    #    plt.plot(np.zeros(len(Bz)),'--k')

    sub2 = plt.subplot(2, 2, 2)
    if np.max(Bx) - np.min(Bx):
        amplitudex = np.max(Bx) - np.min(Bx)
    else:
        amplitudex = 1

    normBx = (Bx - np.mean(Bx)) / amplitudex
    plt.plot(normBy, normBx, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz) / n)
    for i in range(n):
        plt.plot(normBy[i * k], normBx[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    #    plt.plot(normBy[::k],normBx[::k],'ob')
    #    plt.plot(normBy[0],normBx[0],marker='o',
    #                 markersize=6,markerfacecolor='k',
    #                 markeredgewidth=2,markeredgecolor='k') #Begin
    plt.plot(normBy[-1], normBx[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    plt.ylabel('$B_R(B^*_x)$', fontsize=fs)
    # plt.xlabel('$B_T(B^*_z)$',fontsize=fs)
    plt.xlabel('$B_A(B^*_y)$', fontsize=fs)

    # Linha zero
    #    plt.plot(np.zeros(len(By)),'--k')

    # Edição eixos
    #    sub1.yaxis.set_major_locator(plt.NullLocator())
    #    sub1.xaxis.set_major_formatter(plt.NullFormatter())
    sub1.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub1.set_xlim([-1.3, 1.3])
    sub1.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub1.tick_params(axis='x', labelsize=10)
    sub1.set_ylim([-1.3, 1.3])
    sub1.tick_params(axis='y', labelsize=10)
    sub1.grid(True)

    #    sub1.yaxis.set_major_locator(plt.NullLocator())
    sub2.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub2.set_xlim([-1.3, 1.3])
    sub2.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub2.tick_params(axis='x', labelsize=10)
    sub2.set_ylim([-1.3, 1.3])
    sub2.tick_params(axis='y', labelsize=10)
    sub2.grid(True)

    ## PLOT BLAT
    sub3 = plt.subplot(2, 2, 3)
    sub3.plot(BLat, '-k', linewidth=2)
    plt.ylabel('$B_{Lat}$', fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'North'
    t = plt.text(0.8 * len(BLat), 20, ptxt.format(), fontsize=fs)
    ptxt = 'South'
    t = plt.text(0, -70, ptxt.format(), fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(BLat)), '--k')

    ## PLOT BLON
    sub4 = plt.subplot(2, 2, 4)
    plt.plot(BLon, '-k', linewidth=2)
    plt.ylabel('$B_{Lon}$', fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'West'
    t = plt.text(0.8 * len(BLon), 200, ptxt.format(), fontsize=fs)
    ptxt = 'East'
    t = plt.text(0, 20, ptxt.format(), fontsize=fs)
    # Linha 180°
    plt.plot(180 * np.ones(len(BLon)), '--k')

    # sub3.yaxis.set_major_locator(plt.NullLocator())
    sub3.xaxis.set_major_formatter(plt.NullFormatter())
    sub3.set_ylim([-100, 100])
    sub3.yaxis.set_major_locator(plt.MultipleLocator(90))
    sub3.tick_params(axis='y', labelsize=10)
    sub3.grid(False)

    # sub4.yaxis.set_major_locator(plt.NullLocator())
    sub4.xaxis.set_major_formatter(plt.NullFormatter())
    sub4.set_ylim([-10, 370])
    sub4.yaxis.set_major_locator(plt.MultipleLocator(180))
    sub4.tick_params(axis='y', labelsize=10)
    sub4.grid(False)

    return fig


def figura1(modB, Bx, By, Bz, BLat, BLon, info):
    # Figura para comparação ideal vs real
    # com seis subplots dos sinais 2D em linhas
    #
    # info é um vetor de char se necessário

    fs = 14  # fontsize para gráficos

    numfig = 1
    fig = plt.figure(numfig)

    sub1 = plt.subplot(6, 1, 1)
    amplitude = np.max(modB) - np.min(modB)
    normB = (modB - np.mean(modB)) / amplitude
    plt.plot(normB, '-k', linewidth=2, label='$|B|$')
    plt.ylabel('$|B|$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(modB)), '--k')

    sub2 = plt.subplot(6, 1, 2)

    if np.max(Bx) - np.min(Bx):
        amplitude = np.max(Bx) - np.min(Bx)
    else:
        amplitude = 1
    normBx = (Bx - np.mean(Bx)) / amplitude
    plt.plot(normBx, '-k', linewidth=2, label='$B_R(B_x)$')
    plt.ylabel('$B_R(B^*_x)$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Bx)), '--k')

    sub3 = plt.subplot(6, 1, 3)
    if np.max(By) - np.min(By):
        amplitude = np.max(By) - np.min(By)
    else:
        amplitude = 1
    normBy = (By - np.mean(By)) / amplitude
    plt.plot(normBy, '-k', linewidth=2, label='$B_A(B_y)$')
    plt.ylabel('$B_A(B^*_y)$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(By)), '--k')

    sub4 = plt.subplot(6, 1, 4)
    if np.max(Bz) - np.min(Bz):
        amplitude = np.max(Bz) - np.min(Bz)
    else:
        amplitude = 1
    normBz = (Bz - np.mean(Bz)) / amplitude
    plt.plot(normBz, '-k', linewidth=2, label='$B_T(B_z)$')
    plt.ylabel('$B_T(B^*_z)$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Bz)), '--k')

    ## PLOT BLAT
    sub5 = plt.subplot(6, 1, 5)
    sub5.plot(BLat, '-k', linewidth=2)
    plt.ylabel('$B_{Lat}$', fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'North'
    t = plt.text(0.8 * len(BLat), 20, ptxt.format(), fontsize=fs)
    ptxt = 'South'
    t = plt.text(0, -70, ptxt.format(), fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(BLat)), '--k')

    ## PLOT BLON
    sub6 = plt.subplot(6, 1, 6)
    plt.plot(BLon, '-k', linewidth=2)
    plt.ylabel('$B_{Lon}$', fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'West'
    t = plt.text(0.8 * len(BLon), 200, ptxt.format(), fontsize=fs)
    ptxt = 'East'
    t = plt.text(0, 20, ptxt.format(), fontsize=fs)
    # Linha 180°
    plt.plot(180 * np.ones(len(BLon)), '--k')

    # Edição eixos
    #    sub1.yaxis.set_major_locator(plt.NullLocator())
    sub1.xaxis.set_major_formatter(plt.NullFormatter())
    sub1.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub1.set_ylim([-1.3, 1.3])
    sub1.tick_params(axis='y', labelsize=10)
    sub1.grid('off')

    #    sub2.yaxis.set_major_locator(plt.NullLocator())
    sub2.xaxis.set_major_formatter(plt.NullFormatter())
    sub2.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub2.set_ylim([-1.3, 1.3])
    sub2.tick_params(axis='y', labelsize=10)
    sub2.grid('off')

    #    sub3.yaxis.set_major_locator(plt.NullLocator())
    sub3.xaxis.set_major_formatter(plt.NullFormatter())
    sub3.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub3.set_ylim([-1.3, 1.3])
    sub3.tick_params(axis='y', labelsize=10)
    sub3.grid('off')

    #    sub4.yaxis.set_major_locator(plt.NullLocator())
    sub4.xaxis.set_major_formatter(plt.NullFormatter())
    sub4.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub4.set_ylim([-1.3, 1.3])
    sub4.tick_params(axis='y', labelsize=10)
    sub4.grid('off')

    #    sub5.yaxis.set_major_locator(plt.NullLocator())
    sub5.xaxis.set_major_formatter(plt.NullFormatter())
    sub5.set_ylim([-100, 100])
    sub5.yaxis.set_major_locator(plt.MultipleLocator(90))
    sub5.tick_params(axis='y', labelsize=10)
    sub5.grid('off')

    #    sub6.yaxis.set_major_locator(plt.NullLocator())
    sub6.xaxis.set_major_formatter(plt.NullFormatter())
    sub6.set_ylim([-10, 370])
    sub6.yaxis.set_major_locator(plt.MultipleLocator(180))
    sub6.tick_params(axis='y', labelsize=10)
    sub6.grid('off')

    return fig


def figura2(Bx, By, Bz, info):
    # Figura para comparação ideal vs real
    # com dois subplots em linha com planos de máxima e mínima
    #
    # info é um vetor de char se necessário

    fs = 14  # fontsize para gráficos

    numfig = 1
    fig = plt.figure(numfig, figsize=(18, 5))
    plt.subplots_adjust(hspace=0.4)  # Distância entre subplots
    sub1 = plt.subplot(2, 1, 1)
    if np.max(By) - np.min(By):
        amplitudey = np.max(By) - np.min(By)
    else:
        amplitudey = 1

    if np.max(Bz) - np.min(Bz):
        amplitudez = np.max(Bz) - np.min(Bz)
    else:
        amplitudez = 1

    normBy = (By - np.mean(By)) / amplitudey
    normBz = (Bz - np.mean(Bz)) / amplitudez
    # plt.plot(Bz/amplitudez, By/amplitudez,'-k', linewidth=2)
    plt.plot(normBz, normBy, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz) / n)
    for i in range(n):
        plt.plot(normBz[i * k], normBy[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    #    plt.plot(normBz[::k],normBy[::k],'ob')
    #    plt.plot(normBz[0],normBy[0],marker='o',
    #                 markersize=6,markerfacecolor='k',
    #                 markeredgewidth=2,markeredgecolor='k') #Begin
    plt.plot(normBz[-1], normBy[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    # plt.ylabel('$B_R(B^*_x)$',fontsize=fs)
    plt.xlabel('$B_T(B^*_z)$', fontsize=fs)
    plt.ylabel('$B_A(B^*_y)$', fontsize=fs)
    # Linha zero
    #    plt.plot(np.zeros(len(Bz)),'--k')

    sub2 = plt.subplot(2, 1, 2)
    if np.max(Bx) - np.min(Bx):
        amplitudex = np.max(Bx) - np.min(Bx)
    else:
        amplitudex = 1

    normBx = (Bx - np.mean(Bx)) / amplitudex
    plt.plot(normBy, normBx, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz) / n)
    for i in range(n):
        plt.plot(normBy[i * k], normBx[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    #    plt.plot(normBy[::k],normBx[::k],'ob')
    #    plt.plot(normBy[0],normBx[0],marker='o',
    #                 markersize=6,markerfacecolor='k',
    #                 markeredgewidth=2,markeredgecolor='k') #Begin
    plt.plot(normBy[-1], normBx[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    plt.ylabel('$B_R(B^*_x)$', fontsize=fs)
    # plt.xlabel('$B_T(B^*_z)$',fontsize=fs)
    plt.xlabel('$B_A(B^*_y)$', fontsize=fs)

    # Linha zero
    #    plt.plot(np.zeros(len(By)),'--k')

    # Edição eixos
    #    sub1.yaxis.set_major_locator(plt.NullLocator())
    #    sub1.xaxis.set_major_formatter(plt.NullFormatter())
    sub1.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub1.set_xlim([-1.3, 1.3])
    sub1.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub1.tick_params(axis='x', labelsize=10)
    sub1.set_ylim([-1.3, 1.3])
    sub1.tick_params(axis='y', labelsize=10)
    sub1.grid('on')

    #    sub1.yaxis.set_major_locator(plt.NullLocator())
    sub2.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub2.set_xlim([-1.3, 1.3])
    sub2.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub2.tick_params(axis='x', labelsize=10)
    sub2.set_ylim([-1.3, 1.3])
    sub2.tick_params(axis='y', labelsize=10)
    sub2.grid('on')

    return fig


def figura3(modB2, Bx2, By2, Bz2, BLat2, BLon2,
            modB1, Bx1, By1, Bz1, BLat1, BLon1,
            modB3, Bx3, By3, Bz3, BLat3, BLon3,
            vec_theta, vec_phi,
            phi_mva, theta_mva, decision_info):
    # Figura para comparação ideal vs real
    # com seis subplots dos sinais 2D em linhas
    #
    # info é um vetor de char se necessário

    fs = 12  # fontsize para gráficos

    numfig = 1
    fig = plt.figure(numfig)
    sub1 = plt.subplot(6, 3, 1)
    plt.title(r'$\phi_1:$' + str(int(vec_phi[0] * 180 / np.pi)) + '°,' + r'$\theta_1:$' + str(
        int(vec_theta[0] * 180 / np.pi)) + '°',
              fontsize=fs - 1)
    # amplitude = np.max(modB1)-np.min(modB1)
    # normB1 = (modB1-np.mean(modB1))/amplitude
    normB1 = modB1 - np.min(modB1)
    normB1 = normB1 / np.max(normB1)
    plt.plot(normB1, '-k', linewidth=2, label='$|B|$')
    plt.ylabel('$|B|$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(modB1)), '--k')

    sub2 = plt.subplot(6, 3, 2)
    plt.title(
        'Data' + decision_info + r' $\phi:$' + str(int(phi_mva)) + '°,' + r'$\theta:$' + str(int(theta_mva)) + '°',
        fontsize=fs)
    # amplitude = np.max(modB2)-np.min(modB2)
    # normB2 = (modB2-np.mean(modB2))/amplitude
    normB2 = modB2 - np.min(modB2)
    normB2 = normB2 / np.max(normB2)
    plt.plot(normB2, '-k', linewidth=2, label='$|B|$')
    # plt.ylabel('$|B|$',fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(modB2)), '--k')

    sub3 = plt.subplot(6, 3, 3)
    plt.title(r'$\phi_2:$' + str(int(vec_phi[1] * 180 / np.pi)) + '°' + r'$\theta_2:$' + str(
        int(vec_theta[1] * 180 / np.pi)) + '°',
              fontsize=fs - 1)
    # amplitude = np.max(modB3)-np.min(modB3)
    # normB3 = (modB3-np.mean(modB3))/amplitude
    normB3 = modB3 - np.min(modB3)
    normB3 = normB3 / np.max(normB3)
    plt.plot(normB3, '-k', linewidth=2, label='$|B|$')
    # plt.ylabel('$|B|$',fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(modB3)), '--k')

    ####

    sub4 = plt.subplot(6, 3, 4)

    if np.max(Bx1) - np.min(Bx1):
        amplitude = np.max(Bx1) - np.min(Bx1)
    else:
        amplitude = 1
    normBx1 = (Bx1 - np.mean(Bx1)) / amplitude
    plt.plot(normBx1, '-k', linewidth=2, label='$B_R(B_x)$')
    plt.ylabel('$B_R(B^*_x)$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Bx1)), '--k')

    sub5 = plt.subplot(6, 3, 5)

    if np.max(Bx2) - np.min(Bx2):
        amplitude = np.max(Bx2) - np.min(Bx2)
    else:
        amplitude = 1
    normBx2 = (Bx2 - np.mean(Bx2)) / amplitude
    plt.plot(normBx2, '-k', linewidth=2, label='$B_R(B_x)$')
    # plt.ylabel('$B_R(B^*_x)$',fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Bx2)), '--k')

    sub6 = plt.subplot(6, 3, 6)

    if np.max(Bx3) - np.min(Bx3):
        amplitude = np.max(Bx3) - np.min(Bx3)
    else:
        amplitude = 1
    normBx3 = (Bx3 - np.mean(Bx3)) / amplitude
    plt.plot(normBx3, '-k', linewidth=2, label='$B_R(B_x)$')
    # plt.ylabel('$B_R(B^*_x)$',fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Bx3)), '--k')

    ###

    sub7 = plt.subplot(6, 3, 7)
    if np.max(By1) - np.min(By1):
        amplitude = np.max(By1) - np.min(By1)
    else:
        amplitude = 1
    normBy1 = (By1 - np.mean(By1)) / amplitude
    plt.plot(normBy1, '-k', linewidth=2, label='$B_A(B_y)$')
    plt.ylabel('$B_A(B^*_y)$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(By1)), '--k')

    sub8 = plt.subplot(6, 3, 8)
    if np.max(By2) - np.min(By2):
        amplitude = np.max(By2) - np.min(By2)
    else:
        amplitude = 1
    normBy2 = (By2 - np.mean(By2)) / amplitude
    plt.plot(normBy2, '-k', linewidth=2, label='$B_A(B_y)$')
    # plt.ylabel('$B_A(B^*_y)$',fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(By2)), '--k')

    sub9 = plt.subplot(6, 3, 9)
    if np.max(By3) - np.min(By3):
        amplitude = np.max(By3) - np.min(By3)
    else:
        amplitude = 1
    normBy3 = (By3 - np.mean(By3)) / amplitude
    plt.plot(normBy3, '-k', linewidth=2, label='$B_A(B_y)$')
    # plt.ylabel('$B_A(B^*_y)$',fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(By3)), '--k')

    ###

    sub10 = plt.subplot(6, 3, 10)
    if np.max(Bz1) - np.min(Bz1):
        amplitude = np.max(Bz1) - np.min(Bz1)
    else:
        amplitude = 1
    normBz1 = (Bz1 - np.mean(Bz1)) / amplitude
    plt.plot(normBz1, '-k', linewidth=2, label='$B_T(B_z)$')
    plt.ylabel('$B_T(B^*_z)$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Bz1)), '--k')

    sub11 = plt.subplot(6, 3, 11)
    if np.max(Bz2) - np.min(Bz2):
        amplitude = np.max(Bz2) - np.min(Bz2)
    else:
        amplitude = 1
    normBz2 = (Bz2 - np.mean(Bz2)) / amplitude
    plt.plot(normBz2, '-k', linewidth=2, label='$B_T(B_z)$')
    # plt.ylabel('$B_T(B^*_z)$',fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Bz2)), '--k')

    sub12 = plt.subplot(6, 3, 12)
    if np.max(Bz3) - np.min(Bz3):
        amplitude = np.max(Bz3) - np.min(Bz3)
    else:
        amplitude = 1
    normBz3 = (Bz3 - np.mean(Bz3)) / amplitude
    plt.plot(normBz3, '-k', linewidth=2, label='$B_T(B_z)$')
    # plt.ylabel('$B_T(B^*_z)$',fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Bz3)), '--k')

    ## PLOT BLAT
    sub13 = plt.subplot(6, 3, 13)
    sub13.plot(BLat1, '-k', linewidth=2)
    plt.ylabel('$B_{Lat}$', fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'North'
    t = plt.text(0.8 * len(BLat1), 20, ptxt.format(), fontsize=round(fs / 2) + 1)
    ptxt = 'South'
    t = plt.text(0, -70, ptxt.format(), fontsize=round(fs / 2) + 1)
    # Linha zero
    plt.plot(np.zeros(len(BLat1)), '--k')

    ## PLOT BLAT
    sub14 = plt.subplot(6, 3, 14)
    sub14.plot(BLat2, '-k', linewidth=2)
    # plt.ylabel('$B_{Lat}$',fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'North'
    t = plt.text(0.8 * len(BLat2), 20, ptxt.format(), fontsize=round(fs / 2) + 1)
    ptxt = 'South'
    t = plt.text(0, -70, ptxt.format(), fontsize=round(fs / 2) + 1)
    # Linha zero
    plt.plot(np.zeros(len(BLat2)), '--k')

    ## PLOT BLAT
    sub15 = plt.subplot(6, 3, 15)
    sub15.plot(BLat3, '-k', linewidth=2)
    # plt.ylabel('$B_{Lat}$',fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'North'
    t = plt.text(0.8 * len(BLat3), 20, ptxt.format(), fontsize=round(fs / 2) + 1)
    ptxt = 'South'
    t = plt.text(0, -70, ptxt.format(), fontsize=round(fs / 2) + 1)
    # Linha zero
    plt.plot(np.zeros(len(BLat3)), '--k')

    ## PLOT BLON
    sub16 = plt.subplot(6, 3, 16)
    plt.plot(BLon1, '-k', linewidth=2)
    plt.ylabel('$B_{Lon}$', fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'West'
    t = plt.text(0.8 * len(BLon1), 200, ptxt.format(), fontsize=round(fs / 2) + 1)
    ptxt = 'East'
    t = plt.text(0, 20, ptxt.format(), fontsize=round(fs / 2) + 1)
    # Linha 180°
    plt.plot(180 * np.ones(len(BLon1)), '--k')

    ## PLOT BLON
    sub17 = plt.subplot(6, 3, 17)
    plt.plot(BLon2, '-k', linewidth=2)
    # plt.ylabel('$B_{Lon}$',fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'West'
    t = plt.text(0.8 * len(BLon2), 200, ptxt.format(), fontsize=round(fs / 2) + 1)
    ptxt = 'East'
    t = plt.text(0, 20, ptxt.format(), fontsize=round(fs / 2) + 1)
    # Linha 180°
    plt.plot(180 * np.ones(len(BLon2)), '--k')

    ## PLOT BLON
    sub18 = plt.subplot(6, 3, 18)
    plt.plot(BLon3, '-k', linewidth=2)
    # plt.ylabel('$B_{Lon}$',fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'West'
    t = plt.text(0.8 * len(BLon3), 200, ptxt.format(), fontsize=round(fs / 2) + 1)
    ptxt = 'East'
    t = plt.text(0, 20, ptxt.format(), fontsize=round(fs / 2) + 1)
    # Linha 180°
    plt.plot(180 * np.ones(len(BLon3)), '--k')

    ####
    # Edição eixos
    sub1.xaxis.set_major_formatter(plt.NullFormatter())
    sub1.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub1.set_ylim([-1.3, 1.3])
    sub1.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub1.grid('off')

    sub2.xaxis.set_major_formatter(plt.NullFormatter())
    sub2.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub2.set_ylim([-1.3, 1.3])
    sub2.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub2.grid('off')

    sub3.xaxis.set_major_formatter(plt.NullFormatter())
    sub3.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub3.set_ylim([-1.3, 1.3])
    sub3.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub3.grid('off')

    sub4.xaxis.set_major_formatter(plt.NullFormatter())
    sub4.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub4.set_ylim([-1.3, 1.3])
    sub4.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub4.grid('off')

    sub5.xaxis.set_major_formatter(plt.NullFormatter())
    sub5.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub5.set_ylim([-1.3, 1.3])
    sub5.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub5.grid('off')

    sub6.xaxis.set_major_formatter(plt.NullFormatter())
    sub6.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub6.set_ylim([-1.3, 1.3])
    sub6.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub6.grid('off')

    sub7.xaxis.set_major_formatter(plt.NullFormatter())
    sub7.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub7.set_ylim([-1.3, 1.3])
    sub7.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub7.grid('off')

    sub8.xaxis.set_major_formatter(plt.NullFormatter())
    sub8.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub8.set_ylim([-1.3, 1.3])
    sub8.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub8.grid('off')

    sub9.xaxis.set_major_formatter(plt.NullFormatter())
    sub9.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub9.set_ylim([-1.3, 1.3])
    sub9.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub9.grid('off')

    sub10.xaxis.set_major_formatter(plt.NullFormatter())
    sub10.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub10.set_ylim([-1.3, 1.3])
    sub10.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub10.grid('off')

    sub11.xaxis.set_major_formatter(plt.NullFormatter())
    sub11.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub11.set_ylim([-1.3, 1.3])
    sub11.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub11.grid('off')

    sub12.xaxis.set_major_formatter(plt.NullFormatter())
    sub12.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub12.set_ylim([-1.3, 1.3])
    sub12.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub12.grid('off')

    sub13.xaxis.set_major_formatter(plt.NullFormatter())
    sub13.set_ylim([-100, 100])
    sub13.yaxis.set_major_locator(plt.MultipleLocator(90))
    sub13.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub13.grid('off')

    sub14.xaxis.set_major_formatter(plt.NullFormatter())
    sub14.set_ylim([-100, 100])
    sub14.yaxis.set_major_locator(plt.MultipleLocator(90))
    sub14.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub14.grid('off')

    sub15.xaxis.set_major_formatter(plt.NullFormatter())
    sub15.set_ylim([-100, 100])
    sub15.yaxis.set_major_locator(plt.MultipleLocator(90))
    sub15.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub15.grid('off')

    sub16.xaxis.set_major_formatter(plt.NullFormatter())
    sub16.set_ylim([-10, 370])
    sub16.yaxis.set_major_locator(plt.MultipleLocator(180))
    sub16.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub16.grid('off')

    sub17.xaxis.set_major_formatter(plt.NullFormatter())
    sub17.set_ylim([-10, 370])
    sub17.yaxis.set_major_locator(plt.MultipleLocator(180))
    sub17.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub17.grid('off')

    sub18.xaxis.set_major_formatter(plt.NullFormatter())
    sub18.set_ylim([-10, 370])
    sub18.yaxis.set_major_locator(plt.MultipleLocator(180))
    sub18.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub18.grid('off')

    return fig


def figura4(Bx2, By2, Bz2,
            Bx1, By1, Bz1,
            Bx3, By3, Bz3,
            vec_theta, vec_phi,
            theta_mva, phi_mva, info):
    # Figura para comparação ideal vs real
    # com dois subplots em linha com planos de máxima e mínima
    #
    # info é um vetor de char se necessário

    # Matriz de plots
    #
    # sub1 sub2 sub3
    # sub4 sub5 sub6
    #

    fs = 12  # fontsize para gráficos

    numfig = 1
    fig = plt.figure(numfig, figsize=(18, 5))
    plt.subplots_adjust(hspace=0.4)  # Distância entre subplots

    ####
    sub1 = plt.subplot(2, 3, 1)
    plt.title(r'$\phi_1:$' + str(int(vec_phi[0] * 180 / np.pi)) + '°,' + r'$\theta_1:$' + str(
        int(vec_theta[0] * 180 / np.pi)) + '°',
              fontsize=fs - 1)
    if np.max(By1) - np.min(By1):
        amplitudey1 = np.max(By1) - np.min(By1)
    else:
        amplitudey1 = 1

    if np.max(Bz1) - np.min(Bz1):
        amplitudez1 = np.max(Bz1) - np.min(Bz1)
    else:
        amplitudez1 = 1

    normBy1 = (By1 - np.mean(By1)) / amplitudey1
    normBz1 = (Bz1 - np.mean(Bz1)) / amplitudez1
    plt.plot(normBz1, normBy1, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz1) / n)
    for i in range(n):
        plt.plot(normBz1[i * k], normBy1[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    plt.plot(normBz1[-1], normBy1[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    plt.xlabel('$B_T(B^*_z)$', fontsize=fs)
    plt.ylabel('$B_A(B^*_y)$', fontsize=fs)
    # Linha zero

    sub2 = plt.subplot(2, 3, 2)
    plt.title('MVA ' + info + r'$\phi:$' + str(int(phi_mva)) + '°,' + r'$\theta:$' + str(int(theta_mva)) + '°',
              fontsize=fs)

    if np.max(By2) - np.min(By2):
        amplitudey2 = np.max(By2) - np.min(By2)
    else:
        amplitudey2 = 1

    if np.max(Bz2) - np.min(Bz2):
        amplitudez2 = np.max(Bz2) - np.min(Bz2)
    else:
        amplitudez2 = 1

    normBy2 = (By2 - np.mean(By2)) / amplitudey2
    normBz2 = (Bz2 - np.mean(Bz2)) / amplitudez2
    plt.plot(normBz2, normBy2, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz2) / n)
    for i in range(n):
        plt.plot(normBz2[i * k], normBy2[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    plt.plot(normBz2[-1], normBy2[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    plt.xlabel('$B_T(B^*_z)$', fontsize=fs)
    #    plt.ylabel('$B_A(B^*_y)$',fontsize=fs)

    sub3 = plt.subplot(2, 3, 3)
    plt.title(r'$\phi_2:$' + str(int(vec_phi[1] * 180 / np.pi)) + '°,' + r'$\theta_2:$' + str(
        int(vec_theta[1] * 180 / np.pi)) + '°',
              fontsize=fs - 1)

    if np.max(By3) - np.min(By3):
        amplitudey3 = np.max(By3) - np.min(By3)
    else:
        amplitudey3 = 1

    if np.max(Bz3) - np.min(Bz3):
        amplitudez3 = np.max(Bz3) - np.min(Bz3)
    else:
        amplitudez3 = 1

    normBy3 = (By3 - np.mean(By3)) / amplitudey3
    normBz3 = (Bz3 - np.mean(Bz3)) / amplitudez3
    plt.plot(normBz3, normBy3, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz3) / n)
    for i in range(n):
        plt.plot(normBz3[i * k], normBy3[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    plt.plot(normBz3[-1], normBy3[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    plt.xlabel('$B_T(B^*_z)$', fontsize=fs)
    #    plt.ylabel('$B_A(B^*_y)$',fontsize=fs)

    ######

    sub4 = plt.subplot(2, 3, 4)
    if np.max(Bx1) - np.min(Bx1):
        amplitudex1 = np.max(Bx1) - np.min(Bx1)
    else:
        amplitudex1 = 1

    normBx1 = (Bx1 - np.mean(Bx1)) / amplitudex1
    plt.plot(normBy1, normBx1, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz1) / n)
    for i in range(n):
        plt.plot(normBy1[i * k], normBx1[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    plt.plot(normBy1[-1], normBx1[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    plt.ylabel('$B_R(B^*_x)$', fontsize=fs)
    plt.xlabel('$B_A(B^*_y)$', fontsize=fs)

    sub5 = plt.subplot(2, 3, 5)
    if np.max(Bx2) - np.min(Bx2):
        amplitudex2 = np.max(Bx2) - np.min(Bx2)
    else:
        amplitudex2 = 1

    normBx2 = (Bx2 - np.mean(Bx2)) / amplitudex2
    plt.plot(normBy2, normBx2, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz2) / n)
    for i in range(n):
        plt.plot(normBy2[i * k], normBx2[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    plt.plot(normBy2[-1], normBx2[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    #    plt.ylabel('$B_R(B^*_x)$',fontsize=fs)
    plt.xlabel('$B_A(B^*_y)$', fontsize=fs)

    sub6 = plt.subplot(2, 3, 6)
    if np.max(Bx3) - np.min(Bx3):
        amplitudex3 = np.max(Bx3) - np.min(Bx3)
    else:
        amplitudex3 = 1

    normBx3 = (Bx3 - np.mean(Bx3)) / amplitudex3
    plt.plot(normBy3, normBx3, '-k', linewidth=2)

    # Marcações no gráfico
    n = 6  # Número de bolinhas sobre o gráfico
    k = round(len(normBz3) / n)
    for i in range(n):
        plt.plot(normBy3[i * k], normBx3[i * k], alpha=float((n - 1) - i) / (n - 1),
                 marker='o',
                 markersize=8, markerfacecolor='k',
                 markeredgewidth=4, markeredgecolor='k')
    plt.plot(normBy3[-1], normBx3[-1], marker='*',
             markersize=15, markerfacecolor='w',
             markeredgewidth=2, markeredgecolor='k')  # End

    # s    plt.ylabel('$B_R(B^*_x)$',fontsize=fs)
    plt.xlabel('$B_A(B^*_y)$', fontsize=fs)

    # Edição eixos
    sub1.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub1.set_xlim([-1.3, 1.3])
    sub1.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub1.tick_params(axis='x', labelsize=10)
    sub1.set_ylim([-1.3, 1.3])
    sub1.tick_params(axis='y', labelsize=10)
    sub1.grid('on')

    sub2.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub2.set_xlim([-1.3, 1.3])
    sub2.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub2.tick_params(axis='x', labelsize=10)
    sub2.set_ylim([-1.3, 1.3])
    sub2.tick_params(axis='y', labelsize=10)
    sub2.grid('on')

    sub3.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub3.set_xlim([-1.3, 1.3])
    sub3.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub3.tick_params(axis='x', labelsize=10)
    sub3.set_ylim([-1.3, 1.3])
    sub3.tick_params(axis='y', labelsize=10)
    sub3.grid('on')

    #####
    sub4.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub4.set_xlim([-1.3, 1.3])
    sub4.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub4.tick_params(axis='x', labelsize=10)
    sub4.set_ylim([-1.3, 1.3])
    sub4.tick_params(axis='y', labelsize=10)
    sub4.grid('on')

    sub5.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub5.set_xlim([-1.3, 1.3])
    sub5.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub5.tick_params(axis='x', labelsize=10)
    sub5.set_ylim([-1.3, 1.3])
    sub5.tick_params(axis='y', labelsize=10)
    sub5.grid('on')

    sub6.xaxis.set_major_locator(plt.MaxNLocator(3))
    sub6.set_xlim([-1.3, 1.3])
    sub6.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub6.tick_params(axis='x', labelsize=10)
    sub6.set_ylim([-1.3, 1.3])
    sub6.tick_params(axis='y', labelsize=10)
    sub6.grid('on')

    return fig


def figura5(Type, Handedness, eigval, eigvec, phi, theta,
            vec_theta, vec_phi, Pmetric, chi, info_decision):
    # Parâmetros da "tabela"
    fs = 14  # fontsize para gráficos
    vspace = 0.1  # distância entre linhas verticais
    tam = 20
    tamy = 0.5

    numfig = 1
    fig = plt.figure(numfig, figsize=(18, 5))
    # plt.title('MVA and automatic results of experiment')

    # --- Linhas Verticais
    plt.plot([0, 0], [0, tamy], 'k', linewidth=2.0)
    # plt.plot([0.15*tam,0.15*tam],[0,1],'k')
    # plt.plot([0.35*tam,0.35*tam],[0,tamy],'k')
    plt.plot([0.7 * tam, 0.7 * tam], [0, tamy], 'k')
    plt.plot([tam - 1, tam - 1], [0, tamy], 'k', linewidth=2.0)
    # -- Contorno Horizontal
    plt.plot(0 * np.ones(tam), 'k', linewidth=3.0)
    plt.plot(tamy * np.ones(tam), 'k', linewidth=3.0)

    # --- Informações Col 1 e linhas horizontais
    px = 0.01
    py = 0.05
    ptxt = r'$\chi$= ' + str("%3.2f°" % chi)
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    plt.plot((py - 0.05) * np.ones(tam), 'k')
    # ---
    ptxt = r'$\hat x_3$ (GSE)= ' + str("(% 4.3f, % 4.3f, % 4.3f)" % (eigvec[0][2], eigvec[1][2], eigvec[2][2]))
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    plt.plot((py - 0.05) * np.ones(tam), 'k')
    # ---
    ptxt = r'$\hat x_2$ (GSE)= ' + str("(% 4.3f, % 4.3f, % 4.3f)" % (eigvec[0][1], eigvec[1][1], eigvec[2][1]))
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    plt.plot((py - 0.05) * np.ones(tam), 'k')
    # ---
    ptxt = r'$\hat x_1$ (GSE)= ' + str("(% 4.3f, % 4.3f, % 4.3f)" % (eigvec[0][0], eigvec[1][0], eigvec[2][0]))
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    plt.plot((py - 0.05) * np.ones(tam), 'k')
    # ---
    px = 0.3 * tam + 0.01
    ptxt = 'MVA Results' + info_decision
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    plt.plot((py - 0.05) * np.ones(tam), 'k')
    plt.plot([0.35 * tam, 0.35 * tam], [0, py - 0.05], 'k')

    # --- Informações Col 2
    px = 0.35 * tam + 0.01
    py = 0.05
    ptxt = r'$P(\lambda_1,\lambda_2,\lambda_3)$= ' + str('% 2.3f' % (Pmetric))
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    # ---
    ptxt = r'$\lambda_2/\lambda_3$= ' + str("% 3.3f" % (eigval[1] / eigval[2]))
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    # plt.plot((py-0.02)*np.ones(tam),'k')
    # ---
    ptxt = r'$(\lambda_1,\lambda_2,\lambda_3)$= ' + str("(% 4.2f, % 4.2f, % 4.2f)" % (eigval[0], eigval[1], eigval[2]))
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    # plt.plot((py-0.02)*np.ones(tam),'k')
    # ---
    ptxt = r'$(\phi,\theta)$= ' + str("(%3.2f°, % 3.2f°)" % (phi, theta))
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    # plt.plot((py-0.02)*np.ones(tam),'k')

    # --- Informações Col 3
    px = 0.7 * tam + 0.01
    py = 0.05
    ptxt = 'Handedness: ' + Handedness
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    # ---
    ptxt = 'Type: ' + Type
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    # ---
    ptxt = r'$(\phi_2,\theta_2)$= ' + str(
        "(%3i°, % 3i°)" % (int(vec_phi[1] * 180 / np.pi), int(vec_theta[1] * 180 / np.pi)))
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    # ---
    ptxt = r'$(\phi_1,\theta_1)$= ' + str(
        "(%3i°, % 3i°)" % (int(vec_phi[0] * 180 / np.pi), int(vec_theta[0] * 180 / np.pi)))
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    # ---
    px = 0.725 * tam + 0.01
    ptxt = 'Automatic Results'
    py = py + vspace
    t = plt.text(px, py, ptxt.format(), fontsize=fs)
    # plt.plot((py-0.02)*np.ones(tam),'k')

    # Edição eixos
    plt.ylim([0, tamy])
    plt.axis('off')
    plt.grid('off')

    return fig


def figura7(Np, Tp, Ar, Vp, beta, modB, BLat, BLon, hr):
    # Figura para comparação de diferentes

    fs = 12  # fontsize para gráficos

    numfig = 1
    fig = plt.figure(numfig)
    sub1 = plt.subplot(4, 2, 1)
    # plt.title('theta:'+str(int(vec_theta[0]*180/np.pi))+'°,phi:'+str(int(vec_phi[0]*180/np.pi))+'°',
    #          fontsize=fs-1)
    plt.plot(Np, '-k', linewidth=2, label='$H^{+}dens$')
    plt.ylabel('$H^{+}$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Np)), '--k')

    sub2 = plt.subplot(4, 2, 2)
    #    plt.title('ACE data',fontsize=fs)
    plt.plot(beta, '-k', linewidth=2, label='$Beta_p$')
    plt.ylabel('$Beta_p$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(beta)), '--k')

    sub3 = plt.subplot(4, 2, 3)
    plt.plot(Tp, '-k', linewidth=2, label='$H^{+}temp$')
    plt.ylabel('$H^{+}temp$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Tp)), '--k')

    ####

    sub4 = plt.subplot(4, 2, 4)

    normB = modB - np.min(modB)
    normB = normB / np.max(normB)
    plt.plot(normB, '-k', linewidth=2, label='$|B|$')
    plt.ylabel('$|B|$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(modB)), '--k')

    sub5 = plt.subplot(4, 2, 5)
    plt.plot(Ar, '-k', linewidth=2, label='$H^{e}/H^{+}$')
    plt.ylabel('$H^{e}/H^{+}$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Ar)), '--k')

    sub6 = plt.subplot(4, 2, 6)
    ## PLOT BLAT
    sub6.plot(BLat, '-k', linewidth=2)
    plt.ylabel('$B_{Lat}$', fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'North'
    t = plt.text(0.8 * len(BLat), 20, ptxt.format(), fontsize=round(fs / 2) + 1)
    ptxt = 'South'
    t = plt.text(0, -70, ptxt.format(), fontsize=round(fs / 2) + 1)
    # Linha zero
    plt.plot(np.zeros(len(BLat)), '--k')

    ###

    sub7 = plt.subplot(4, 2, 7)
    plt.plot(hr, Vp, '-k', linewidth=2, label='$H^{+}spped$')
    plt.ylabel('$H^{+}spped$', fontsize=fs)
    # Linha zero
    plt.plot(np.zeros(len(Vp)), '--k')

    sub8 = plt.subplot(4, 2, 8)
    ## PLOT BLON
    plt.plot(hr, BLon, '-k', linewidth=2)
    plt.ylabel('$B_{Lon}$', fontsize=fs)

    # Anotação sobre o gráfico
    ptxt = 'West'
    t = plt.text(0.8 * len(BLon), 200, ptxt.format(), fontsize=round(fs / 2) + 1)
    ptxt = 'East'
    t = plt.text(0, 20, ptxt.format(), fontsize=round(fs / 2) + 1)
    # Linha 180°
    plt.plot(180 * np.ones(len(BLon)), '--k')

    ####
    # Edição eixos
    Num_x = 8  # numero de linhas verticais em x
    sub1.xaxis.set_major_formatter(plt.NullFormatter())
    sub1.xaxis.set_major_locator(plt.MaxNLocator(Num_x))
    sub1.yaxis.set_major_locator(plt.MaxNLocator(3))
    # sub1.set_ylim([-1.3,1.3])
    sub1.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub1.grid('on')

    sub2.xaxis.set_major_formatter(plt.NullFormatter())
    sub2.xaxis.set_major_locator(plt.MaxNLocator(Num_x))
    sub2.yaxis.set_major_locator(plt.MaxNLocator(3))
    # sub2.set_ylim([-1.3,1.3])
    sub2.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub2.grid('on')

    sub3.xaxis.set_major_formatter(plt.NullFormatter())
    sub3.xaxis.set_major_locator(plt.MaxNLocator(Num_x))
    sub3.yaxis.set_major_locator(plt.MaxNLocator(3))
    # sub3.set_ylim([-1.3,1.3])
    sub3.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub3.grid('on')

    sub4.xaxis.set_major_formatter(plt.NullFormatter())
    sub4.xaxis.set_major_locator(plt.MaxNLocator(Num_x))
    sub4.yaxis.set_major_locator(plt.MaxNLocator(3))
    sub4.set_ylim([-0.3, 1.2])
    sub4.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub4.grid('on')

    sub5.xaxis.set_major_formatter(plt.NullFormatter())
    sub5.xaxis.set_major_locator(plt.MaxNLocator(Num_x))
    sub5.yaxis.set_major_locator(plt.MaxNLocator(3))
    # sub5.set_ylim([-1.3,1.3])
    sub5.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub5.grid('on')

    sub6.xaxis.set_major_formatter(plt.NullFormatter())
    sub6.xaxis.set_major_locator(plt.MaxNLocator(Num_x))
    sub6.set_ylim([-100, 100])
    sub6.yaxis.set_major_locator(plt.MultipleLocator(90))
    sub6.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub6.grid('on')

    for label in sub7.get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
    sub7.xaxis.set_major_locator(plt.MaxNLocator(Num_x))

    sub7.yaxis.set_major_locator(plt.MaxNLocator(3))
    # sub7.set_ylim([-1.3,1.3])
    sub7.tick_params(axis='y', labelsize=round(fs / 2) + 2)
    sub7.tick_params(axis='x', labelsize=round(fs / 2) + 2)
    sub7.grid('on')

    for label in sub8.get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
        # print(label)
    sub8.xaxis.set_major_locator(plt.MaxNLocator(Num_x))

    # sub8.xaxis.set_major_formatter(plt.NullFormatter())
    sub8.set_ylim([-10, 370])
    sub8.yaxis.set_major_locator(plt.MultipleLocator(180))
    sub8.tick_params(axis='y', labelsize=round(fs / 2) + 1)
    sub8.tick_params(axis='x', labelsize=round(fs / 2) + 1)
    sub8.grid('on')

    return fig
