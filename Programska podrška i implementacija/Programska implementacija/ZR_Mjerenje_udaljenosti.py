# -*- coding: cp1250 -*-
import cv2, random, os, re, sys
import cPickle as pickle
import numpy as np

'''
SVEUČILIŠTE U ZAGREBU
FAKULTET ELEKTROTEHNIKE I RAČUNARSTVA

Mentor: lzv.prof.dr.sc. Siniša Šegvić

ZAVRŠNI RAD br. 3667
Naslova rada: Mjerenje udaljenosti u ravnini ceste kamerom postavljenom na vozilu
Student: Josip Milić


Zagreb, lipanj 2014.
'''

#preusmjeravanje u folder

print os.getcwd()

folder = "Testni primjeri\\"

#duljina stranice kvadrata
#kalibracijskog uzorka

duljinaKvadrataCM = 6

#dimenzije demonstracijskog objekta
#pomocu kojeg se mjeri pogreska izracuna

#vrijedi za primjer GOPR2203:

'''
duzinaObjekta = 23.7
visinaObjekta = 16.35
'''

#papir A4 formata (za npr. GOPR2252):

duzinaObjekta = 29.7
visinaObjekta = 21.0

povrsinaObjekta = duzinaObjekta*visinaObjekta

# Broj unutarnjih kutova uzorka u retku i stupcu
#brojUnutarnjihKutova = (7,5)
brojUnutarnjihKutova = [5,3]

# Funkcija za uvecanje slike
def prikaziUvecano(koordinate):
    # Polovica broja piksela duzine isjecka slike6
    brojPiksela = 50
    # Ako se odabere tocka ciji ce isjecak prijeci rubove slike, tocka se korigira
    # na nacin da rub isjecka lezi na rubu slike
    if koordinate[0] - brojPiksela < 0:
        koordinate[0] += abs(koordinate[0] - brojPiksela)
    if koordinate[1] - brojPiksela < 0:
        koordinate[1] += abs(koordinate[1] - brojPiksela)
    if koordinate[0] + brojPiksela >= slika.shape[1]:
        koordinate[0] -= abs(koordinate[0]+brojPiksela-slika.shape[1])+1
    if koordinate[1] + brojPiksela >= slika.shape[0]:
        koordinate[1] -= abs(koordinate[1]+brojPiksela-slika.shape[0])+1

    slikaUvecana = slika[koordinate[1]-brojPiksela:koordinate[1]+brojPiksela+1, koordinate[0]-brojPiksela:koordinate[0]+brojPiksela+1]
    slikaUvecana_orig = slikaUvecana.copy()

    # Povecanje isjecka slike
    povecanje = 5
    slikaUvecana = cv2.resize(slikaUvecana, (0,0), fx=povecanje, fy=povecanje) 
    slikaUvecana_orig2 = slikaUvecana.copy()
    centar = [slikaUvecana.shape[1]/2,slikaUvecana.shape[0]/2]
    cv2.line(slikaUvecana, (centar[0],0), (centar[0],slikaUvecana.shape[0]), [0,255,255],1,cv2.CV_AA )
    cv2.line(slikaUvecana, (0,centar[1]), (slikaUvecana.shape[1],centar[1]), [0,255,255],1,cv2.CV_AA )
    enter = 0
    while(1):
        cv2.imshow('Pomoc pri odabiru', slikaUvecana)
        k = cv2.waitKey(33)
        # Tipka Enter
        if k == 13:
            cv2.destroyWindow('Pomoc pri odabiru')
            enter = 1
            break
        # Tipka Escape
        if k == 27:
            cv2.destroyWindow('Pomoc pri odabiru')
            break

        # Zoom
        # Tipka +
        if k == 43:
            if povecanje <15:
                povecanje +=1
                slikaUvecana = slikaUvecana_orig.copy()
                slikaUvecana = cv2.resize(slikaUvecana, (0,0), fx=povecanje, fy=povecanje) 
                centar = [slikaUvecana.shape[1]/2,slikaUvecana.shape[0]/2]
                slikaUvecana_orig2 = slikaUvecana.copy()
                cv2.line(slikaUvecana, (centar[0],0), (centar[0],slikaUvecana.shape[0]), [0,255,255],1,cv2.CV_AA )
                cv2.line(slikaUvecana, (0,centar[1]), (slikaUvecana.shape[1],centar[1]), [0,255,255],1,cv2.CV_AA )
        # Tipka -
        if k == 45:
            if povecanje >1:
                povecanje -=1
                slikaUvecana = slikaUvecana_orig.copy()
                slikaUvecana = cv2.resize(slikaUvecana, (0,0), fx=povecanje, fy=povecanje) 
                centar = [slikaUvecana.shape[1]/2,slikaUvecana.shape[0]/2]
                slikaUvecana_orig2 = slikaUvecana.copy()
                cv2.line(slikaUvecana, (centar[0],0), (centar[0],slikaUvecana.shape[0]), [0,255,255],1,cv2.CV_AA )
                cv2.line(slikaUvecana, (0,centar[1]), (slikaUvecana.shape[1],centar[1]), [0,255,255],1,cv2.CV_AA )

        # Navigacija
        # Lijevo
        if k==2424832 or k==ord('a'):
            slikaUvecana = slikaUvecana_orig2.copy()
            centar[0] -= 1
            cv2.line(slikaUvecana, (centar[0],0), (centar[0],slikaUvecana.shape[0]), [0,255,255],1,cv2.CV_AA )
            cv2.line(slikaUvecana, (0,centar[1]), (slikaUvecana.shape[1],centar[1]), [0,255,255],1,cv2.CV_AA )
        # Gore
        if k==2490368 or k==ord('w'):
            slikaUvecana = slikaUvecana_orig2.copy()
            centar[1] -= 1
            cv2.line(slikaUvecana, (centar[0],0), (centar[0],slikaUvecana.shape[0]), [0,255,255],1,cv2.CV_AA )
            cv2.line(slikaUvecana, (0,centar[1]), (slikaUvecana.shape[1],centar[1]), [0,255,255],1,cv2.CV_AA )
        # Desno
        if k==2555904 or k==ord('d'):
            slikaUvecana = slikaUvecana_orig2.copy()
            centar[0] += 1
            cv2.line(slikaUvecana, (centar[0],0), (centar[0],slikaUvecana.shape[0]), [0,255,255],1,cv2.CV_AA )
            cv2.line(slikaUvecana, (0,centar[1]), (slikaUvecana.shape[1],centar[1]), [0,255,255],1,cv2.CV_AA )
        # Dolje
        if k==2621440 or k==ord('s'):
            slikaUvecana = slikaUvecana_orig2.copy()
            centar[1] += 1
            cv2.line(slikaUvecana, (centar[0],0), (centar[0],slikaUvecana.shape[0]), [0,255,255],1,cv2.CV_AA )
            cv2.line(slikaUvecana, (0,centar[1]), (slikaUvecana.shape[1],centar[1]), [0,255,255],1,cv2.CV_AA )


    # Ako je pritisnuta tipka Enter spremi koordinate, inace ne
    if (enter):
        pomakX = centar[0]-slikaUvecana.shape[1]/2
        pomakY = centar[1]-slikaUvecana.shape[0]/2
        koordinate[0] += float(pomakX)/povecanje
        koordinate[1] += float(pomakY)/povecanje
        return koordinate

# Funkcija za odabir tocaka
def odabirTocaka(event,x,y,flags,param):
    global tockaSlovo
    global slika
    global korekcijaUkljucena
    if event == cv2.EVENT_LBUTTONDOWN:
        slikaSiva = cv2.cvtColor(slikaOriginal, cv2.COLOR_BGR2GRAY)
        if pomocOdabir or korekcijaUkljucena:
            if korekcijaUkljucena and (listaKoordinata):     
                for i in range(len(listaKoordinata)):
                    xk,yk = listaKoordinata[i]
                    if x in range(int(round(xk))-10,int(round(xk))+10) and y in range(int(round(yk))-10,int(round(yk))+10):
                        x = int(round(xk))
                        y = int(round(yk))
                        indexKorekcije = i
                        slika = slikaOriginal.copy()
                        cv2.circle(slika,(int(round(x)),int(round(y))),1,[0,255,0],1)
                        break
            koordinate = prikaziUvecano([x,y])

            
            if koordinate:
                if korekcijaUkljucena:
                    listaKoordinata[indexKorekcije] = koordinate
                    tockaSlovo = 65
                    slika = slikaOriginal.copy()
                    for i in range(len(listaKoordinata)):
                        x,y = listaKoordinata[i]
                        cv2.circle(slika,(int(round(x)),int(round(y))),1,[0,255,0],1)
                        tockaSlovo += 1
                    korekcijaUkljucena = 0

                else:
                    listaKoordinata.append(koordinate)
                    cv2.circle(slika,(int(round(koordinate[0])),int(round(koordinate[1]))),1,(0,255,0),1)

                    #ZA SLIKU
                    tockaMatrica = np.array([[[x,y]]],dtype = np.float32)
                    matricaKamere = pickle.load(open( folder+"matricaKamere.dat", "rb" ))
                    koeficijentiIzoblicenja = pickle.load(open( folder+"koeficijentiIzoblicenja.dat", "rb" ))
                    tockaPopravljeneMatrica = cv2.undistortPoints(tockaMatrica,matricaKamere,koeficijentiIzoblicenja,P=matricaKamere)
                    koorBez = int(round(float( tockaPopravljeneMatrica[0][0][0]))),int(round(float(tockaPopravljeneMatrica[0][0][1])))
                    cv2.circle(slika,koorBez,1,(0,255,255),1)
                    #ZA SLIKU
                    
                    #cv2.circle(slika,(int(round(koordinate[0])),int(round(koordinate[1]))),5,(0,255,0),1)
                    #cv2.line(slika,(0,int(round(koordinate[1]))),(slika.shape[1],int(round(koordinate[1]))),(0,255,100),1)
                    #cv2.putText(slika,chr(tockaSlovo),(int(round(koordinate[0]))+5,int(round(koordinate[1]))+5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),5,cv2.CV_AA)
                    #cv2.putText(slika,chr(tockaSlovo),(int(round(koordinate[0]))+4,int(round(koordinate[1]))+4),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                    tockaSlovo += 1
 
        else:
            listaKoordinata.append([x,y])
            cv2.circle(slika,(x,y),1,(0,255,0),1)
            cv2.circle(slika,(x,y),5,(0,255,0),4)

            #ZA SLIKU
            tockaMatrica = np.array([[[x,y]]],dtype = np.float32)
            matricaKamere = pickle.load(open( folder+"matricaKamere.dat", "rb" ))
            koeficijentiIzoblicenja = pickle.load(open( folder+"koeficijentiIzoblicenja.dat", "rb" ))
            tockaPopravljeneMatrica = cv2.undistortPoints(tockaMatrica,matricaKamere,koeficijentiIzoblicenja,P=matricaKamere)
            koorBez = int(round(float( tockaPopravljeneMatrica[0][0][0]))),int(round(float(tockaPopravljeneMatrica[0][0][1])))
            if koorBez[0]>0 and koorBez[1]>0:
                #print 'Sa:',(x,y),'Bez:',koorBez
                cv2.circle(slika,koorBez,1,(0,255,255),1)
                cv2.circle(slika,koorBez,5,(0,255,255),4)
            else:
                pass
                #print 'Sa:',(x,y),'Bez:',koorBez
            #ZA SLIKU
            
            #cv2.line(slika,(0,y),(slika.shape[1],y),(0,255,100),1)
            #cv2.putText(slika,chr(tockaSlovo),(x+1,y+1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),5,cv2.CV_AA)
            #cv2.putText(slika,chr(tockaSlovo),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
            tockaSlovo += 1

def makniIzoblicenje(listaKoordinata,slikaMicanje):
    global slika
    if (listaKoordinata):
        tockeMatrica = np.array([listaKoordinata],dtype = np.float32)
    # Ako postoje spremljeni parametri kamere, koristi ih za micanje izoblicanje tocaka
    try:
        matricaKamere = pickle.load(open( folder+"matricaKamere.dat", "rb" ))
        koeficijentiIzoblicenja = pickle.load(open( folder+"koeficijentiIzoblicenja.dat", "rb" ))
    except:
        # Ako ne postoje, izvadi ih iz datoteke s podacima kamere
        try:
            podaciKamere = open(folder+"out_camera_data.yml","r").read()
            podaciMatrica = [(int(x.split(' ')[1].rstrip()), int(x.split(' ')[-1])) for x in re.findall(r'rows: \d+\s.+cols: \d+',podaciKamere)]
            indexi = [m.start(0) for m in re.finditer('\[', podaciKamere)]+[m.start(0) for m in re.finditer('\]', podaciKamere)]

            vrijednostiMatrica = [podaciKamere[indexi[0]+1:indexi[2]].replace(' ','').replace('\n',''),podaciKamere[indexi[1]+1:indexi[3]].replace(' ','').replace('\n','')]
            vrijednostiMatrica = [[float(vrijednost) for vrijednost in vrijednosti.split(',')] for vrijednosti in vrijednostiMatrica]

            matricaKamere = np.array(vrijednostiMatrica[0],dtype=np.float32).reshape(podaciMatrica[0][0],podaciMatrica[0][1])
            koeficijentiIzoblicenja = np.array(vrijednostiMatrica[1],dtype=np.float32).reshape(podaciMatrica[1][0],podaciMatrica[1][1])

            pickle.dump( matricaKamere, open( folder+"matricaKamere.dat", "wb" ) )
            pickle.dump( koeficijentiIzoblicenja, open( folder+"koeficijentiIzoblicenja.dat", "wb" ) )
        except:
            sys.stdout.write("Pogreska: Nedostaje datoteka 'out_camera_data.yml' koja je potrebna za micanje izoblicenja!\n")
            return
    if (listaKoordinata):
        tockePopravljeneMatrica = cv2.undistortPoints(tockeMatrica,matricaKamere,koeficijentiIzoblicenja,P=matricaKamere)
        listaKoordinata = [[float(koor[0]),float(koor[1])] for koor in tockePopravljeneMatrica[0]]
    if (len(slikaMicanje)):
        return cv2.undistort(slikaMicanje,matricaKamere,koeficijentiIzoblicenja)
    else:
        return listaKoordinata

def korekcijaTocaka(event,x,y,flags,param):
    global slikaTransf
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(len(listaKoordinataTransf[0])):
            xk,yk = listaKoordinata[0][i]

            if x in range(int(round(xk))-10,int(round(xk))+10) or y in range(int(round(yk))-10,int(round(yk))+10):
                cv2.circle(slikaTransf,(xk,yk),2,[255,255,0])
                cv2.circle(slikaTransf,(xk,yk),6,[255,255,0])



def transformiraj(listaKoordinata):
    global slika
    global rjecnikIzracuna
    global estimacijaUkljucena

    try:
        if not estimacijaUkljucena:
            sys.stdout.write('Izvor podataka kalibracijskog uzorka: '+folder+"podaciUzorka.dat"+'\n')
        podaciUzorka,brojUnutarnjihKutova = pickle.load(open( folder+"podaciUzorka.dat", "rb"))

        # Gornji lijevi vrh uzorka se stavlja u (0,0) koordinatnog sustava
        # pomakom tog vrha za pomakPiksela se moze vidjeti veci dio transformirane slike

        pomakPiksela = 2500
        # Duzina kvadrata uzorka prikazana brojem piksela
        brojPikselaKvadrat = 20

        # Vrhovi uzorka u koordinatnom sustavu
        vrhoviUzorkaKS = podaciUzorka[:]
        vrhoviUzorkaKS[0] = [pomakPiksela,pomakPiksela]
        vrhoviUzorkaKS[1] = [pomakPiksela+(brojUnutarnjihKutova[0]-1)*brojPikselaKvadrat,pomakPiksela]
        vrhoviUzorkaKS[2] = [pomakPiksela+(brojUnutarnjihKutova[0]-1)*brojPikselaKvadrat,pomakPiksela+(brojUnutarnjihKutova[1]-1)*brojPikselaKvadrat]
        vrhoviUzorkaKS[3] = [pomakPiksela,pomakPiksela+(brojUnutarnjihKutova[1]-1)*brojPikselaKvadrat]

        # Pretvorba u oblik matrice kako bi se izracunala homografija
        vrhoviUzorkaMatrica = np.array(podaciUzorka,dtype = np.float32)
        vrhoviUzorkaKSMatrica = np.array(vrhoviUzorkaKS,dtype = np.float32)

        transfMatrica = cv2.findHomography(vrhoviUzorkaKSMatrica,vrhoviUzorkaMatrica)[0]
        transfMatrica = np.linalg.inv(transfMatrica)

        listaKoordinataMatrica = np.array(listaKoordinata,dtype = np.float32)
        listaKoordinataMatrica = np.array([listaKoordinataMatrica])

        podaciUzorkaMatrica = np.array(podaciUzorka,dtype = np.float32)
        podaciUzorkaMatrica = np.array([podaciUzorkaMatrica])
        listaKoordinataTransf = cv2.perspectiveTransform(listaKoordinataMatrica,transfMatrica)
        
        podaciUzorkaTransf = cv2.perspectiveTransform(podaciUzorkaMatrica,transfMatrica)
        # Prikaz transformirane slike - nebitno za izracur povrsine
        slika = slikaOriginal.copy()
        try:
            matricaKamere = pickle.load(open( folder+"matricaKamere.dat", "rb" ))
            koeficijentiIzoblicenja = pickle.load(open( folder+"koeficijentiIzoblicenja.dat", "rb" ))
            slika = cv2.undistort(slika,matricaKamere,koeficijentiIzoblicenja)
        except:
            sys.stdout.write("Nedostaju parametri kamere, izoblicenje slike se nije maknulo!\n")
        slikaTransf=cv2.warpPerspective(slika,transfMatrica,(3*slika.shape[1],3*slika.shape[0]),0,cv2.INTER_LINEAR)
        slika = slikaOriginal.copy()
        koordinataTransfXNajveca,koordinataTransfYNajveca = listaKoordinataTransf[0][0]
        koordinataTransfXNajmanja,koordinataTransfYNajmanja = listaKoordinataTransf[0][0]
        tockaSlovo = 65
        
        for i in range(len(podaciUzorkaTransf[0])):
            x,y = podaciUzorkaTransf[0][i]
            cv2.circle(slikaTransf,(x,y),2,[0,255,255])
            cv2.circle(slikaTransf,(x,y),6,[0,255,255])
        

        for i in range(len(listaKoordinataTransf[0])):
            x,y = listaKoordinataTransf[0][i]

            cv2.circle(slikaTransf,(x,y),2,[0,255,0])
            cv2.circle(slikaTransf,(x,y),6,[0,255,0])
            
            cv2.putText(slikaTransf,chr(tockaSlovo),(int(round(x)-8),int(round(y))+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),5,cv2.CV_AA)
            cv2.putText(slikaTransf,chr(tockaSlovo),(int(round(x)-7),int(round(y))+19),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

            if x > koordinataTransfXNajveca: koordinataTransfXNajveca = x
            if y > koordinataTransfYNajveca: koordinataTransfYNajveca = y
            if x < koordinataTransfXNajmanja: koordinataTransfXNajmanja = x
            if y < koordinataTransfYNajmanja: koordinataTransfYNajmanja = y
            tockaSlovo+=1
        tockaSlovo = 65
        #cv2.imwrite('.'.join(lokacijaSlike.split('.')[:-1])+" - Transform - Novi."+lokacijaSlike.split('.')[-1],slikaTransf)
        brojPikselaTransf = 50
        slikaTransf = slikaTransf[koordinataTransfYNajmanja-brojPikselaTransf:koordinataTransfYNajveca+brojPikselaTransf,koordinataTransfXNajmanja-brojPikselaTransf:koordinataTransfXNajveca+brojPikselaTransf]
        #cv2.namedWindow('Transformirani dio slike',cv2.WINDOW_NORMAL)

        # Povrsina uzorka na transformiranoj slici u pikselima
        povrsinaUzorkaSlika = cv2.contourArea(vrhoviUzorkaKSMatrica)
        # Stvarna povrsina uzorka
        povrsinaUzorka = (brojUnutarnjihKutova[0]-1)*(brojUnutarnjihKutova[1]-1)*(duljinaKvadrataCM**2)

        # Povrsina objekta na transformiranoj slici u pikselima
        povrsinaObjektaSlika = cv2.contourArea(listaKoordinataTransf)
        povrsinaObjektaIzm = povrsinaObjektaSlika/(povrsinaUzorkaSlika/povrsinaUzorka)
        pogreska = -(1-povrsinaObjektaIzm/povrsinaObjekta)*100
        rjecnikIzracuna = {}
        rjecnikIzracuna['P'] = povrsinaObjektaIzm

        if not estimacijaUkljucena:
            sys.stdout.write('Povrsina predloska: '+str(povrsinaUzorka)+' cm^2 \n')
            sys.stdout.write('Izmjerena povrsina objekta:  '+str(round(povrsinaObjektaIzm,2))+' cm^2 \n')
            sys.stdout.write('Stvarna povrsina objekta: '+str(duzinaObjekta)+' cm * '+str(visinaObjekta)+' cm = '+str(povrsinaObjekta)+' cm^2\n')
            sys.stdout.write('Razlika povrsina objekta: '+str(round(povrsinaObjektaIzm-povrsinaObjekta,2))+' cm^2 \n')
            sys.stdout.write('Pogreska izracuna povrsine: '+str(pogreska)+' %\n (tocnost: '+str(100-abs(pogreska))+' % )')
            sys.stdout.write('\n')
            sys.stdout.write('\tStvarne udaljenosti izmedu oznacenih vrhova:\n')
            sys.stdout.write('\tDuzina = '+str(duzinaObjekta)+' cm\n')
            sys.stdout.write('\tVisina = '+str(visinaObjekta)+' cm\n')
            sys.stdout.write('\tIzmjerene udaljenosti izmedu oznacenih vrhova:\n')
        for i in range(len(listaKoordinataTransf[0])):
            if (i!=len(listaKoordinataTransf[0])-1):
                for j in range(i+1,len(listaKoordinataTransf[0])):
                    x1,y1=float(listaKoordinataTransf[0][i][0]),float(listaKoordinataTransf[0][i][1])
                    x2,y2=float(listaKoordinataTransf[0][j][0]),float(listaKoordinataTransf[0][j][1])
                    d = float(np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)))
                    if not estimacijaUkljucena:
                        izmjerenaDuzina = round(d*(float(duljinaKvadrataCM)/brojPikselaKvadrat),2)
                        sys.stdout.write('\t|'+chr(tockaSlovo+i)+chr(tockaSlovo+j)+'| = '+str(izmjerenaDuzina)+' cm')
                        if abs(izmjerenaDuzina-duzinaObjekta) < 4:
                            pogreskaDuzina = -(1-izmjerenaDuzina/duzinaObjekta)*100
                            sys.stdout.write('\t D Razlika: '+str(izmjerenaDuzina-duzinaObjekta)+' cm (tocnost: '+str(round(100-abs(pogreskaDuzina),2))+' % )\n')
                        elif abs(izmjerenaDuzina-visinaObjekta) < 4:
                            pogreskaDuzina = -(1-izmjerenaDuzina/visinaObjekta)*100
                            sys.stdout.write('\t V Razlika: '+str(izmjerenaDuzina-visinaObjekta)+' cm (tocnost: '+str(round(100-abs(pogreskaDuzina),2))+' % )\n')
                        else:
                            sys.stdout.write('\n')
                    rjecnikIzracuna[chr(tockaSlovo+i)+chr(tockaSlovo+j)] = d*(float(duljinaKvadrataCM)/brojPikselaKvadrat)
        if estimacijaUkljucena:
            return

        # Uvecan prikaz dijela transformirane slike u kojoj se nalaze odabrane tocke
        povecanje = 1
        slikaTransfUvecana = cv2.resize(slikaTransf, (0,0), fx=povecanje, fy=povecanje) 
        slikaTransfUvecana_orig = slikaTransfUvecana.copy()

        while(1):
                cv2.imshow('Transformirani dio slike', slikaTransfUvecana)
                k = cv2.waitKey(33)
                # Tipka Enter
                if k == 13:
                    cv2.destroyWindow('Transformirani dio slike')
                    enter = 1
                    break
                # Tipka Escape
                if k == 27:
                    cv2.destroyWindow('Transformirani dio slike')
                    break
                if k == ord('e'):
                    if estimacijaUkljucena:
                        estimacijaUkljucena = 0
                        sys.stdout.write('Estimacija duljina i povrsine: iskljucena\n')
                    else:
                        estimacijaUkljucena = 1
                        sys.stdout.write('Estimacija duljina i povrsine: ukljucena\n')

                # Zoom
                # Tipka +
                if k == 43:
                    if povecanje <15 and povecanje > 0.5:
                        povecanje +=1
                        slikaTransfUvecana = slikaTransfUvecana_orig.copy()
                        slikaTransfUvecana = cv2.resize(slikaTransfUvecana, (0,0), fx=povecanje, fy=povecanje) 
                    if povecanje <= 0.5:
                        povecanje *= float(2)
                        slikaTransfUvecana = slikaTransfUvecana_orig.copy()
                        slikaTransfUvecana = cv2.resize(slikaTransfUvecana, (0,0), fx=povecanje, fy=povecanje) 
                # Tipka -
                if k == 45:
                    if povecanje >1:
                        povecanje -=1
                        slikaTransfUvecana = slikaTransfUvecana_orig.copy()
                        slikaTransfUvecana = cv2.resize(slikaTransfUvecana, (0,0), fx=povecanje, fy=povecanje)    
                    if povecanje < 2 and povecanje > 0.01:
                        povecanje /= float(2)
                        slikaTransfUvecana = slikaTransfUvecana_orig.copy()
                        slikaTransfUvecana = cv2.resize(slikaTransfUvecana, (0,0), fx=povecanje, fy=povecanje)

    
    except:
        sys.stdout.write('Doslo je do pogreske!\n')
    return listaKoordinata


def izracunajPovrsinu(listaKoordinata):
    global detekcijaUkljucena

    if not (detekcijaUkljucena):
        listaKoordinata = makniIzoblicenje(listaKoordinata,[])
    '''
    for l in listaKoordinata:
        cv2.circle(slika,(int(round(l[0])),int(round(l[1]))),1,(0,255,255),1)
        cv2.circle(slika,(int(round(l[0])),int(round(l[1]))),5,(0,255,255),1)
    '''
    listaKoordinata = transformiraj(listaKoordinata)


def detekcijaUzorka():
    global slika
    global slikaOriginal
    global detekcijaUkljucena
    sys.stdout.write('Pokrenuta je automatska detekcija uzorka...\n')
    brojUnutarnjihKutovaDetekcija = brojUnutarnjihKutova 
    # Automatska detekcija uzorka
    slikaBezIzoblicenja = makniIzoblicenje(0,slikaOriginal)
    while(brojUnutarnjihKutovaDetekcija[0] >= 3 and brojUnutarnjihKutovaDetekcija[1] >= 3):
        sys.stdout.write('Traze se '+str(brojUnutarnjihKutovaDetekcija[0]-1)+' kvadrata u jednom retku i '+str(brojUnutarnjihKutovaDetekcija[1]-1)+' kvadrata u jednom stupcu...\n')
        podaciUzorkaDetekcija = cv2.findChessboardCorners(slikaBezIzoblicenja, tuple(brojUnutarnjihKutovaDetekcija), flags = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        #podaciUzorkaDetekcija = cv2.findChessboardCorners(slikaBezIzoblicenja, brojUnutarnjihKutova)
        if (podaciUzorkaDetekcija[0]):
            sys.stdout.write('Kalibracijski predlozak je automatski pronaden!\n')
            podaciUzorkaDetekcija = podaciUzorkaDetekcija[1]
            slikaSiva = cv2.cvtColor(slikaBezIzoblicenja, cv2.COLOR_BGR2GRAY)
            cv2.cornerSubPix(slikaSiva,podaciUzorkaDetekcija,(detekcijaPr,detekcijaPr),(-1,-1),(cv2.TERM_CRITERIA_MAX_ITER +cv2.TERM_CRITERIA_EPS, 30, 0.1))
            vrhoviUzorkaDetekcija =  [
                                      [podaciUzorkaDetekcija[0][0][0],podaciUzorkaDetekcija[0][0][1]],
                                      [podaciUzorkaDetekcija[brojUnutarnjihKutovaDetekcija[0]-1][0][0],podaciUzorkaDetekcija[brojUnutarnjihKutovaDetekcija[0]-1][0][1]],
                                      [podaciUzorkaDetekcija[brojUnutarnjihKutovaDetekcija[0]*brojUnutarnjihKutovaDetekcija[1]-1][0][0],podaciUzorkaDetekcija[brojUnutarnjihKutovaDetekcija[0]*brojUnutarnjihKutovaDetekcija[1]-1][0][1]],
                                      [podaciUzorkaDetekcija[(brojUnutarnjihKutovaDetekcija[1]-1)*brojUnutarnjihKutovaDetekcija[0]][0][0],podaciUzorkaDetekcija[(brojUnutarnjihKutovaDetekcija[1]-1)*brojUnutarnjihKutovaDetekcija[0]][0][1]]
                                     ]
            # Identificiranje vrhova (za slucaj da nisu odabrani na pravilan nacin)
            vrhoviUzorkaDetekcija = sorted([koor for koor in vrhoviUzorkaDetekcija])
            if vrhoviUzorkaDetekcija[0][1] > vrhoviUzorkaDetekcija[1][1]:
                donjiLijeviKut = vrhoviUzorkaDetekcija[0]
                gornjiLijeviKut = vrhoviUzorkaDetekcija[1]
            else:
                donjiLijeviKut = vrhoviUzorkaDetekcija[1]
                gornjiLijeviKut = vrhoviUzorkaDetekcija[0]
            if vrhoviUzorkaDetekcija[2][1] > vrhoviUzorkaDetekcija[3][1]:
                donjiDesniKut = vrhoviUzorkaDetekcija[2]
                gornjiDesniKut = vrhoviUzorkaDetekcija[3]
            else:
                donjiDesniKut = vrhoviUzorkaDetekcija[3]
                gornjiDesniKut = vrhoviUzorkaDetekcija[2]
            vrhoviUzorkaDetekcija = [gornjiLijeviKut,gornjiDesniKut,donjiDesniKut,donjiLijeviKut]

            podaciUzorka = [vrhoviUzorkaDetekcija,brojUnutarnjihKutovaDetekcija]
            pickle.dump( podaciUzorka, open( folder+"podaciUzorka.dat", "wb" ) )
            izracunajPovrsinu(vrhoviUzorkaDetekcija)
            detekcijaUkljucena = 0
            return
        else:     
            if brojUnutarnjihKutovaDetekcija[1] > 3:
                brojUnutarnjihKutovaDetekcija[1] -= 1
            else:
                brojUnutarnjihKutovaDetekcija[1] = brojUnutarnjihKutova[1]
                brojUnutarnjihKutovaDetekcija[0] -= 1
                if brojUnutarnjihKutovaDetekcija[0] == brojUnutarnjihKutovaDetekcija[1] and brojUnutarnjihKutovaDetekcija[1] > 3:
                    brojUnutarnjihKutovaDetekcija[1] -= 1
                    brojUnutarnjihKutovaDetekcija[1] = brojUnutarnjihKutova [1]



    slikaOriginal = slika.copy()
    slika = slikaBezIzoblicenja.copy()
    sys.stdout.write('Kalibracijski predlozak nije automatski pronaden!\nOdaberite vrhove predloska na sljedeci nacin:\n')
    sys.stdout.write('\t1. vrh: gornji lijevi vrh prvog crnog kvadrata\n')
    sys.stdout.write('\t2. vrh: gornji desni vrh zadnjeg kvadrata u prvom retku\n')
    sys.stdout.write('\t3. vrh: donji desni vrh zadnjeg kvadrata u zadnjem retku\n')
    sys.stdout.write('\t4. vrh: donji lijevi vrh prvog kvadrata u zadnjem retku\n')

def odaberiFolderSliku(folder,sklopka):
    folderOriginal = folder
    while(sklopka):
        sys.stdout.write('Folderi:\n\n')
        listaFoldera = [imeFoldera for imeFoldera in os.listdir(folder+'\\') if len(imeFoldera.split('.'))==1]
        for i in range(len(listaFoldera)):
            sys.stdout.write('\t'+str(i+1)+'. '+listaFoldera[i]+'\n')
        try:
            ulaz = raw_input('Upisite broj foldera (za izlaz pritisnite enter) : ')
            brojFoldera = int(ulaz)-1
            folder = folder.split('\\')[0]+'\\'+listaFoldera[brojFoldera]+"\\"
            break
        except:
            if not(ulaz):
                break
            else:
                sys.stdout.write('Upisali ste nepostojeci folder! Pokusajte ponovno.\nZa izlaz pritisnite tipku enter\n\n')
    if (sklopka) and not(ulaz):
        sys.exit(0)
    while(1):
        listaSlika = [imeSlike for imeSlike in os.listdir(folder) if 'slika' in imeSlike and len(imeSlike.split(' '))==1]
        sys.stdout.write('\nSlike u folderu '+folder+' :\n\n')
        for i in range(len(listaSlika)):
            sys.stdout.write('\t'+str(i+1)+'. '+listaSlika[i]+'\n')
        try:
            ulaz = raw_input('Upisite broj slike (za ispis liste foldera pritisnite enter) : ')
            brojSlike = int(ulaz)-1
            lokacijaSlike = folder+listaSlika[brojSlike]
            break
        except:
            if not(ulaz):
                lokacijaSlike = odaberiFolderSliku(folder.split('\\')[0],1)
                break
            else:
                sys.stdout.write('Upisali ste nepostojecu sliku! Pokusajte ponovno.\nZa prikaz liste foldera pritisnite tipku enter\n\n')
    return lokacijaSlike



listaKoordinata = []
pomocOdabir = 1
tockaSlovo = 65
detekcijaPr = 2
detekcijaUkljucena = 0
pitajBrojKvadrata = 0
estimacijaUkljucena = 0
korekcijaUkljucena = 0
listaKoordinataPrethodna = []
lokacijaSlike = odaberiFolderSliku(folder.split('\\')[0]+'\\',1)
folder = '\\'.join(lokacijaSlike.split('\\')[:-1])+'\\'
print lokacijaSlike
slika = cv2.imread(lokacijaSlike)
slikaOriginal = slika.copy()

cv2.namedWindow('Odabiranje',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Odabiranje',odabirTocaka)


# Prikaz slike
while(1):
    cv2.imshow('Odabiranje', slika)
    k = cv2.waitKey(33)

    # Objasnjenje precica (hotkeys)
    if k == ord('h'):
        try:
            sys.stdout.write(open(folder.split('\\')[0]+'\\help_(precice).txt','r').read())
        except:
            sys.stdout.write('Nedostaje datoteka help_precice.txt!\n')

    # Odabirom tipke p se onemogucava/omogucava pomoc:
    # uvecanje dijela slike i ravnala koja se mogu pomicati
    if k == ord('p'):
        pomocOdabir = 1-pomocOdabir
        if pomocOdabir:
            print "Pomoc omogucena!"
        else:
            print "Pomoc onemogucena!"
    # Tipka enter
    if k == 13:
        tockaSlovo = 65
        if (detekcijaUkljucena):
            cv2.destroyAllWindows()
            brojUnutarnjihKutova = [int(raw_input('Upisite broj kvadrata u jednom retku izmedu oznacenih vrhova: '))+1,int(raw_input('Upisite broj kvadrata u jednom stupcu izmedu oznacenih vrhova: '))+1]
            cv2.namedWindow('Odabiranje',cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Odabiranje',odabirTocaka)
            listaKoordinataSubPix = np.array([listaKoordinata],dtype=np.float32)
            slikaBezIzoblicenja = makniIzoblicenje(0,slikaOriginal)
            slikaSiva = cv2.cvtColor(slikaBezIzoblicenja, cv2.COLOR_BGR2GRAY)
            detekcijaPr = 2
            cv2.cornerSubPix(slikaSiva,listaKoordinataSubPix,(detekcijaPr,detekcijaPr),(-1,-1),(cv2.TERM_CRITERIA_MAX_ITER +cv2.TERM_CRITERIA_EPS, 30, 0.1))
            print listaKoordinataSubPix
            listaKoordinata = [[koor[0],koor[1]] for koor in listaKoordinataSubPix[0]]
            print 'listaKoordinata',listaKoordinata
            podaciUzorka = [listaKoordinata,brojUnutarnjihKutova]
            pickle.dump( podaciUzorka, open( folder+"podaciUzorka.dat", "wb" ) )
        if listaKoordinata:
            '''
            if len(listaKoordinata) <= 2:
                sys.stdout.write('Potrebno je odabrati najmanje 3 tocke!\n')
            else:
            '''
            listaKoordinataPrethodna = listaKoordinata[:]
            izracunajPovrsinu(listaKoordinata)
            if (estimacijaUkljucena):    
                sys.stdout.write('Estimacija duljina i povrsine: pokrenuta...\n')
                br=0
                listaIzracunaIzmjerena = [rjecnikIzracuna['P']]
                listaIzracunaMjere = ['Povrsina']
                for izracun in rjecnikIzracuna:
                    if izracun != 'P':
                        listaIzracunaIzmjerena.append(rjecnikIzracuna[izracun])
                        listaIzracunaMjere.append(izracun)
                listaIzracuna = len(rjecnikIzracuna)*[0]
                #sys.stdout.write('\n\t'+str(listaKoordinata)+'\n')

                '''
                listaIzracuna.append(rjecnikIzracuna['P'])
                for izracun in sorted(rjecnikIzracuna):
                    if izracun != 'P':
                        listaIzracuna.append(rjecnikIzracuna[izracun])
                '''
                pomakPiksela = 2
                for i in range(len(listaKoordinata)):
                    for j in range(len(listaKoordinata[i])):
                        for k in range(pomakPiksela):
                            ispisPomak = 'ulijevo'
                            if j==1:
                                ispisPomak = 'gore'
                            sys.stdout.write('\tKoordinata '+chr(65+i)+' se pomice za jedan piksel '+ispisPomak+'\n')
                            listaKoordinata[i][j]-=1
                            #sys.stdout.write('\n'+str(listaKoordinata)+'\n')
                            izracunajPovrsinu(listaKoordinata)
                            listaIzracuna[0]+=rjecnikIzracuna['P']
                            br+=1
                            brojPom = 1
                            for izracun in sorted(rjecnikIzracuna):
                                if izracun != 'P':
                                    listaIzracuna[brojPom]+=rjecnikIzracuna[izracun]
                                    brojPom+=1
                        
                        listaKoordinata[i][j] += pomakPiksela+1
                        for k in range(pomakPiksela):
                            ispisPomak = 'udesno'
                            if j==1:
                                ispisPomak = 'dolje'
                            sys.stdout.write('\tKoordinata '+chr(65+i)+' se pomice za jedan piksel '+ispisPomak+'\n')
                            listaKoordinata[i][j]+=1
                            #sys.stdout.write('\n'+str(listaKoordinata)+'\n')
                            izracunajPovrsinu(listaKoordinata)
                            listaIzracuna[0]+=rjecnikIzracuna['P']
                            br+=1
                            brojPom = 1
                            for izracun in sorted(rjecnikIzracuna):
                                if izracun != 'P':
                                    listaIzracuna[brojPom]+=rjecnikIzracuna[izracun]
                                    brojPom+=1
                        
                        listaKoordinata[i][j] -= pomakPiksela
                        sys.stdout.write('\tKoordinata '+chr(65+i)+' se vraca na pocetnu poziciju\n')
                        #sys.stdout.write('\n\t'+str(listaKoordinata)+'\n')
                sys.stdout.write('Estimacija duljina i povrsine: zavrsena!\n')
                for i in range(len(listaIzracuna)):
                    listaIzracuna[i]/=br
                    if i==0:
                        sys.stdout.write('Povrsina = '+str(listaIzracunaIzmjerena[i])+' +- '+str(abs(float(-(1-listaIzracuna[i]/listaIzracunaIzmjerena[i]))))+' cm^2\n')
                    else:
                        sys.stdout.write('|'+listaIzracunaMjere[i]+'| = '+str(listaIzracunaIzmjerena[i])+' +- '+str(abs(float(-(1-listaIzracuna[i]/listaIzracunaIzmjerena[i]))))+' cm\n')
                estimacijaUkljucena = 0

                #izracunajPovrsinu(listaKoordinata)
            listaKoordinata = []
        if (detekcijaUkljucena):
            slika = slikaOriginal.copy()
            detekcijaUkljucena = 0
    if k== ord('c'):
        if korekcijaUkljucena:
            korekcijaUkljucena = 0
            sys.stdout.write('\nKorekcija tocke je onemogucena!\n')
        else:
            korekcijaUkljucena = 1
            sys.stdout.write('\nKorekcija tocke je omogucena!\n')
    if k== ord('d'):
        sys.stdout.write('\nPokrenuta je detekcija uzorka!\n')
        slika = slikaOriginal.copy()
        detekcijaUkljucena = 1
        detekcijaUzorka()
    # Resetiranje slike
    if k== ord('r'):
        tockaSlovo = 65
        slika = slikaOriginal.copy()
        listaKoordinata = []
    # Promjena slike
    if k== ord('s'):
        cv2.destroyAllWindows()
        folder = '\\'.join(lokacijaSlike.split('\\')[:-1])+'\\'
        lokacijaSlike = odaberiFolderSliku(folder,0)
        folder = '\\'.join(lokacijaSlike.split('\\')[:-1])+'\\'
        slika = cv2.imread(lokacijaSlike)
        slikaOriginal = slika.copy()
        cv2.namedWindow('Odabiranje',cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Odabiranje',odabirTocaka)
    if k== ord('f') or k==27:
        cv2.destroyAllWindows()
        folder = lokacijaSlike.split('\\')[0]+'\\'
        lokacijaSlike = odaberiFolderSliku(folder,1)
        folder = '\\'.join(lokacijaSlike.split('\\')[:-1])+'\\'
        slika = cv2.imread(lokacijaSlike)
        slikaOriginal = slika.copy()
        cv2.namedWindow('Odabiranje',cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Odabiranje',odabirTocaka)
    if k== ord('e'):
        if not(estimacijaUkljucena):
            sys.stdout.write('Estimacija duljina i povrsine: ukljucena\n')
            estimacijaUkljucena = 1
        else:
            sys.stdout.write('Estimacija duljina i povrsine: iskljucena\n')
            estimacijaUkljucena = 0
    if k== ord('k'):
        cv2.destroyAllWindows()
        sys.stdout.write('\nTrenutno je postavljeno da se maksimalno trazi\n'+str(brojUnutarnjihKutova[0]-1)+' kvadrata u jednom retku i '+str(brojUnutarnjihKutova[1]-1)+' kvadrata u jednom stupcu.\n')
        sys.stdout.write('Upisite nove vrijednosti ili pritisnite enter za potvrdu trenutnih.\n')
        try:
            brojUnutarnjihKutova = [int(raw_input('Upisite broj kvadrata u jednom retku: '))+1,int(raw_input('Upisite broj kvadrata u jednom stupcu: '))+1]
        except:
            sys.stdout.write('Potvrdene trenutne vrijednosti\n')
        cv2.namedWindow('Odabiranje',cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Odabiranje',odabirTocaka)
    if k==ord('l'):
        if listaKoordinataPrethodna:
            sys.stdout.write('Prikazane su i odabrane prethodno izabrane tocke.\n')
            tockaSlovo = 65
            for i in range(len(listaKoordinataPrethodna)):
                x,y = listaKoordinataPrethodna[i]

                cv2.circle(slika,(int(round(x)),int(round(y))),2,[25,255,125])
                cv2.circle(slika,(int(round(x)),int(round(y))),6,[25,255,125])
                
                cv2.putText(slika,chr(tockaSlovo),(int(round(x)-8),int(round(y))+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),5,cv2.CV_AA)
                cv2.putText(slika,chr(tockaSlovo),(int(round(x)-7),int(round(y))+19),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                tockaSlovo+=1
            listaKoordinata = listaKoordinataPrethodna[:]
    if k== ord('v'):
        cv2.destroyAllWindows()
        sys.stdout.write('\nTrenutne vrijednosti dimenzija objekta kojem se mjeri povrsina\nDuzina: '+str(duzinaObjekta)+'\nVisina: '+str(visinaObjekta)+' \n')
        sys.stdout.write('Upisite nove vrijednosti ili pritisnite enter za potvrdu trenutnih.\n')
        dimenzijeObjektaOriginalne = duzinaObjekta,visinaObjekta
        try:
            duzinaObjekta,visinaObjekta = float(raw_input('Upisite duzinu objekta: ')),float(raw_input('Upisite visinu objekta: '))
            povrsinaObjekta = duzinaObjekta*visinaObjekta
        except:
            duzinaObjekta,visinaObjekta = dimenzijeObjektaOriginalne
            sys.stdout.write('Potvrdene trenutne vrijednosti\n')
        cv2.namedWindow('Odabiranje',cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Odabiranje',odabirTocaka)
    if k== ord('b'):
        primjerx = 0
        primjery = 0
        pomocOdabir = 1-pomocOdabir
        while(1):
            primjerx+=100
            primjery = 0
            if primjerx>slika.shape[1]:
                break
            while(1):
                primjery+=100
                if primjery>slika.shape[0]:
                    break
                
                odabirTocaka(cv2.EVENT_LBUTTONDOWN,primjerx,primjery,0,0)
        pomocOdabir = 1-pomocOdabir
            
        



