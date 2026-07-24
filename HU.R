p = c(0.5,0.2,0.3)
sum(p)
o = rmultinom(1,10,prob=p)
o
o = rmultinom(1,100,p)
o
likelihood=dmultinom(o,prob=p)
likelihood
dmultinom(o,prob=c(0.4,0.3,0.3))
dmultinom(o,prob=c(0.57,0.19,0.24))
o
dmultinom(c(50,20,30),prob=p)
210*.4^4*.6^6
dbinom(4,10,p=0.4)

(factorial(10)/(factorial(4)^2*2))*(0.4)^4*(0.5)^4*(0.1)^2
dmultinom(c(4,4,2),10,c(.4,.5,.1))  

dmultinom(c(4,4,2),10,c(.4,.5,.1),log=T)


