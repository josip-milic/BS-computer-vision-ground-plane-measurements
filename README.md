Sveučilište u Zagrebu<br>
<a href="http://www.fer.unizg.hr">Fakultet elektrotehnike i računarstva</a>

# <a href="http://www.fer.unizg.hr/predmet/">Završni rad</a>

2014./2015.

# Tema rada: 
Ovim radom opisani su postupci dobivanja modela stvaranja slike, kalibracije 
kamere i preslikavanja točaka slike u ravninu ceste. Tim postupcima ostvarilo se
mjerenje udaljenosti u ravnini ceste uz pomoć slike dobivene kamerom postavljenom
na vozilu. Za mjerenje potrebna je relativno jeftina kamera i tanka podloga s
kalibracijskim uzorkom. Postupak kalibracije kamere implementiran je uz pomoć
C++-a, a ravninsko preslikavanje uz pomoć Pythona. Metode računalnog vida,
potrebne za ovaj rad, implementirane su uz pomoć OpenCV biblioteke. Korišteni
NumPy paket omogućio je stvaranje i korištenje matrica. Eksperimentalni rezultati
pokazali su relativno vrlo precizno mjerenje udaljenosti u ravnini ceste korištenjem
opisanih postupaka. Veće pogreške mjerenja uočile su se kod točaka u ravnini ceste
koje su se nalazile na relativno velikoj udaljenosti od kamere.   


(c) 2015 Josip Milić
<br>

<img src="https://github.com/josip-milic/BS-computer-vision-ground-plane-measurements/blob/master/Slike/panel_kalibracija.png">
<br/>
<img src="https://github.com/josip-milic/BS-computer-vision-ground-plane-measurements/blob/master/Slike/micanje_izoblicenja.png">
<br/>
<img src="https://github.com/josip-milic/BS-computer-vision-ground-plane-measurements/blob/master/Slike/panel_homografija.png">
<br/>
<img src="https://github.com/josip-milic/BS-computer-vision-ground-plane-measurements/blob/master/Slike/cesta_1.png">
<br/>
<img src="https://github.com/josip-milic/BS-computer-vision-ground-plane-measurements/blob/master/Slike/pomoc_pri_odabiru_1.png">
<br/>
<img src="https://github.com/josip-milic/BS-computer-vision-ground-plane-measurements/blob/master/Slike/transform.png">
<br/>

<pre>
Slika: cesta_1.png
Povrsina objekta: 23.7 cm * 16.35 cm = 387.49 cm^2
Izmjerena povrsina objekta: 4306.15562104 piksela (387.554005893cm^2)
Povrsina predloska: 9600.0 piksela (864.0cm^2)
Pogreska: 0.0165153481441 %
</pre>

###Više informacija i upute o korištenju u <a href="https://github.com/josip-milic/BS-computer-vision-ground-plane-measurements/blob/master/Zavr%C5%A1ni%20rad/milic_bs_rad.pdf">radu</a>.

