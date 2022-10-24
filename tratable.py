import pandas as pd
import funcoes
import figuras
import numpy as np
import scipy.signal as ss

arq = open('name1.csv', 'r')

dados2 = []
dados1 = []
dados3 = []
dados4 = []
for line in arq:
    dados1.append(line[:-1].split(' ', 17))

    line2 = line[:-1].split(' ', 17)
    if line2[0] > '2021/07/06':
        dados3.append(line2)
        if len(line2[0]) == 10:
            dados2.append(line2)

dados = pd.DataFrame(dados2)
da = (dados.iloc[:, [3, 4, 6, 7]])
dadoss = pd.DataFrame(da)
dadoss.to_csv("dados2.csv", encoding='utf-8')
# print(dadoss)
da[3] = pd.to_datetime(da[3], format='%Y/%m/%d')
da[6] = pd.to_datetime(da[6], format='%Y/%m/%d')
#da[4] = round(da[4],2)
da['anoini'] = pd.DatetimeIndex(da[3]).year
da['mesini'] = pd.DatetimeIndex(da[3]).month
da['diaini'] = pd.DatetimeIndex(da[3]).day
da['anofim'] = pd.DatetimeIndex(da[6]).year
da['mesfim'] = pd.DatetimeIndex(da[6]).month
da['diainfim'] = pd.DatetimeIndex(da[6]).day
#da['horario'] = pd.Timestamp(da[4])
table1 = (da.iloc[:, [4, 5, 6, 1, 7, 8, 9, 3]])
tables = pd.DataFrame(table1)




def main0(table, arqdados, opt_plot):
            # abertura do arquivo
            arq2 = open( arqdados, 'r')


            # Extrai dados linha a linha do arquivo arqdados

            dados = []
            inicio = 0
            arqcol = []

            for line in arq2:
                if inicio == 1:  # Pular cabeçalho
                    line2 = line[:-1].split(' ')
                    while '' in line2:
                        del line2[line2.index('')]  # filtro de espaços

                    dados.append(line2)  # Copia dados em kip separando colunas

                elif line != 'BEGIN DATA\n':
                    arqcols = line[:-1].split(' ')
                if line == 'BEGIN DATA\n':
                    inicio = 1
            arq2.close()

            Bmagcalc = 0

            # Identificação de Colunas e mensagem ERROR
            arqcols = np.array(arqcols)

            if not ('year' in arqcols):
                return ['', "Input file without 'year' data!"]
            elif not ('day' in arqcols):
                return ['', "Input file without 'day' data!"]
            elif not ('hr' in arqcols):
                return ['', "Input file without 'hr' data!"]
            #    elif not ('min' in arqcols):
            #        return['',"Input file without 'min' data!"]
            #    elif not ('sec' in arqcols):
            #        return['',"Input file without 'sec' data!"]
            elif not ('Bmag' in arqcols):
                # return['',"Input file without 'Bmag' data!"]
                Bmagcalc = 1
            elif not ('Bgse_x' in arqcols or 'B_gse_x' in arqcols):
                return ['', "Input file without 'Bgse_x' data!"]
            elif not ('Bgse_y' in arqcols or 'B_gse_y' in arqcols):
                return ['', "Input file without 'Bgse_y' data!"]
            elif not ('Bgse_z' in arqcols or 'B_gse_z' in arqcols):
                return ['', "Input file without 'Bgse_z' data!"]

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
            if float(dados[x_begin][yearcol]) > start_year:
                return ['', "Input file without 'start year' entry!"]
            if float(dados[x_begin][daycol]) > start_day:
                return ['', "Input file without 'start day' entry!"]
            if float(dados[x_begin][hrcol]) > start_hour:
                return ['', "Input file without 'start hour' entry!"]

            test = float(dados[x_begin][yearcol])  # year

            while test < start_year:
                x_begin = x_begin + 1
                if x_begin > lendados:
                    return ['', "Input file without 'start year' entry!"]
                test = float(dados[x_begin][yearcol])  # year

            test = float(dados[x_begin][daycol])  # day

            while test < start_day:
                x_begin = x_begin + 1
                if x_begin > lendados:
                    return ['', "Input file without 'start day' entry!"]
                test = float(dados[x_begin][daycol])  # day

            test = float(dados[x_begin][hrcol])  # hr

            while test < start_hour:
                x_begin = x_begin + 1
                if x_begin > lendados:
                    return ['', "Input file without 'start hour' entry!"]
                test = float(dados[x_begin][hrcol])  # hr

            print('inicio= ' + str(dados[x_begin][yearcol]) + str(dados[x_begin][daycol]) + str(dados[x_begin][hrcol]))
            # Localiza o fim do evento (termina na linha x_end)
            x_end = x_begin + 1
            if x_end > lendados:
                return ['', "Input file without 'start' entry!"]
            test = float(dados[x_end][yearcol])  # year

            while test < end_year:
                x_end = x_end + 1
                if x_end > lendados:
                    return ['', "Input file without 'end year' entry!"]
                test = float(dados[x_end][yearcol])  # year

            test = float(dados[x_end][daycol])  # day

            while test < end_day:
                x_end = x_end + 1
                if x_end > lendados:
                    return ['', "Input file without 'end day' entry!"]
                test = float(dados[x_end][daycol])  # day

            test = float(dados[x_end][hrcol])  # hr

            while test < end_hour:
                x_end = x_end + 1
                if x_end > lendados:
                    return ['', "Input file without 'end hour' entry!"]

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


            BL, BM, BN, eigval, eigvec, phi, theta, chi = funcoes.funcao2(Bx, By, Bz)
            theta = round(theta) * np.pi / 180
            phi = round(phi) * np.pi / 180
            BLat, BLon = funcoes.funcao3(Bx, By, Bz)
            kernel_size = round(len(BLat) / 10 - 1)
            if kernel_size % 2 == 0:
                kernel_size = kernel_size - 1
            fBLat = ss.medfilt(BLat, kernel_size)
            fBLon = ss.medfilt(BLon, kernel_size)
            tipo, hand = funcoes.funcaotype(fBLat, fBLon, theta)
            Pmetric = round((funcoes.metric(eigval)), 3)
            vec_phi, vec_theta, vec_hand = funcoes.anglelist(round(phi * 180 / np.pi), round(theta * 180 / np.pi), tipo)
            phi = funcoes.calc_longitudinal(eigvec)
            phi_mva = round((phi[1]), 3)
            theta = funcoes.calc_latitudinal(eigvec)
            theta_mva = round((theta[1]), 3)
            inicio_fim = start_year
            pltBx_id = []
            pltBy_id = []
            pltBz_id = []
            r = int(len(modB) / 50) - 1  # rescaling
            pltmodB_real = modB[::r]
            pltBx_real = Bx[::r]
            pltBy_real = By[::r]
            pltBz_real = Bz[::r]
            pltBL = BL[::r]
            pltBM = BM[::r]
            pltBN = BN[::r]
            lamb2 = abs(eigval[1] / eigval[0])
            lamb3 = abs(eigval[2] / eigval[0])
            razao = round((lamb2/lamb3), 3)
            if  opt_plot == 'dadosca':
                #dadoscara1 = (funcoes.save_hodograms(pltBy_id[0],pltBz_id[0],pltBx_id[0],pltBM,pltBL,pltBN,pltBy_id[1],pltBz_id[1],pltBx_id[1],pltBx_real, pltBy_real, pltBz_real,vec_theta,vec_phi,theta_mva,phi_mva,table,tipo,hand,eigval, eigvec,Pmetric,chi))
                dadoscara = figuras.figura5(tipo, hand, eigval, eigvec, phi_mva, theta_mva, vec_theta, vec_phi, Pmetric, chi, '')
                carac = []
                carac.append(dadoscara)
                angulophietheta = phi_mva, theta_mva
                print(angulophietheta)

                tipoi = tipo
                with open('dados2.csv', encoding='utf-8') as ar:
                    tabela12 = pd.read_csv(ar, delimiter=',')
                    #tabela12.insert(tipo,start_year,str(angulophietheta),razao,Pmetric, round((chi),3))
                    #tabela12.append(tipo,start_year,str(angulophietheta),razao,Pmetric, round((chi),3))
                    #print(tabela12)
                    #d8 = []
                    d7 = pd.DataFrame(tabela12, columns=['3', '4', '6', '7'])

                    d7.insert(0, "1", tipo, allow_duplicates=True)
                    d7.insert(1, "2", start_year, allow_duplicates=True)
                    d7.insert(5, "8", str(angulophietheta), allow_duplicates=True)
                    d7.insert(6, "12", razao, allow_duplicates=True)
                    d7.insert(7, "13", Pmetric, allow_duplicates=True)
                    d7.insert(8, "14", round((chi), 3), allow_duplicates=True)
                    d7.insert(9, "9", 'NOT DATA', allow_duplicates=True)
                    d7.insert(11, "10", 'NOT DATA', allow_duplicates=True)
                    d7.insert(12, "11", 'NOT DATA', allow_duplicates=True)
                   # print(d7)
                    #d7_insert =[[tipo,start_year,str(angulophietheta),razao,Pmetric, round((chi),3)]]
                    #dfA = d7.iloc[:,0,1,2,3]
                    #dfB = d7.loc[d7['3']==start_day]
                    #df = dfA.append(d7_insert).append(dfB).reset_index(drop=True)
                    #print(df)

                    dados0= [[tipo,start_year,str(angulophietheta),razao,Pmetric, round((chi),3)]]
                    df2 = pd.DataFrame((dados0), columns=['1', '2', '8', '12', '13','14'])
                   # with open('noal.csv', mode='w') as bas:
                    #    bas.write(str(tipo) +';'+ str(start_year) +';'+ str(angulophietheta) +';'+ str(razao) +';'+ str(Pmetric) +';'+ str(round((chi),3)))









        #        with open('Planilhaeventos.csv', encoding='utf-8') as arqui:
        #            tabela1 = pd.read_csv(arqui, delimiter=';')
        #        df2 = pd.DataFrame(tabela1, columns=['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14'])
        #    m = pd.concat([df2, d7], ignore_index=True, axis=0)
        #    df1 = pd.DataFrame(m, columns=['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14'])
        #    df1.to_excel("d6.xls")
        #    df1.to_csv("plandados.csv")
        #    print(df1)
            print(tipo)
            return dados0

for i in range(15):
    print(i)
#while i < 8:
 #   i += 0
    a = tables.loc[i, 'anoini']
    b = tables.loc[i, 'mesini']
    c = tables.loc[i, 'diaini']
    d = tables.loc[i, 4]
    e = tables.loc[i, 'anofim']
    f = tables.loc[i, 'mesfim']
    g = tables.loc[i, 'diainfim']
    h = tables.loc[i, 7]
    table = [str(a), str(b), str(c), str(d[0:2]), str(e), str(f), str(g), str(h[0:2])]
    print(table)
#retorna as funcões
    if a == 2021:
        arqdados = 'ACE_MAG_Data (1).txt'
        opt_plot = 'dadosca'
        TESTE = main0(table, arqdados, opt_plot)

    if a == 2022:
        arqdados = 'ACE_MAG_Data 2022.txt'
        opt_plot = 'dadosca'
        TESTE = main0(table, arqdados, opt_plot)
    print(TESTE)

    if i == 0:
        df2 = pd.DataFrame(TESTE,  columns=['1', '2', '8', '12', '13','14'])
    if i == 1:
         df21 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13','14'])
    if i == 2:
        df22 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13','14'])

    if i == 3:
        df23 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13','14'])

    if i == 4:
        df24 = pd.DataFrame(TESTE, columns=['1', '2', '8', '12', '13', '14'])

    if i == 5:
        df25 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])

    if i == 6:
        df26 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])
    if i == 7:
        df27 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])
    if i == 8:
        df28 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])
    if i == 9:
        df29 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])
    if i == 10:
        df30 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])
    if i == 11:
        df31 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])
    if i == 12:
        df32 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])
    if i == 13:
        df33 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])
    if i == 14:
        df34 = pd.DataFrame((TESTE), columns=['1', '2', '8', '12', '13', '14'])
        m = pd.concat([df2, df21, df22, df23, df24, df25, df26, df27, df28, df29, df30, df31, df32, df33, df34], ignore_index=True, axis=0)
        df10 = pd.DataFrame(m, columns=['1', '2', '8', '12', '13', '14'])
        print(m)
        with open('dados2.csv', encoding='utf-8') as ar:
            tabela12 = pd.read_csv(ar, delimiter=',')
            d7 = pd.DataFrame(tabela12, columns=['3', '4', '6', '7'])
            d7.insert(0, "9", '*', allow_duplicates=True)
            d7.insert(1, "10", '*', allow_duplicates=True)
            d7.insert(2, "11", '*', allow_duplicates=True)
        with open('Planilhaeventos.csv', encoding='utf-8') as arqui:
            tabela1 = pd.read_csv(arqui, delimiter=';')
            dadosantigos = pd.DataFrame(tabela1, columns=['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14'])
            n = pd.concat([d7,df10], axis=1)
            dadosatua = pd.DataFrame(n, columns=['3', '4', '6', '7','9', '10','11', '1', '2', '8', '12', '13', '14'])
            print(dadosatua)
            dadosatua.to_csv('nada.csv')
            o = pd.concat([dadosantigos, dadosatua], ignore_index=True, axis=0)
            dadosfim = pd.DataFrame(o, columns=['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14'])
            dadosfim.to_excel("d6.xls")
            dadosfim.to_csv("plandados.csv")
            #print(dadosfim)