print("--------DETERMINANTE DE MATRIZ--------\n")

mtx_size = int(input("Digite o tamanho da MATRIZ: "))

matrix = []
mtx_column = mtx_size
mtx_line = mtx_size


# CRIAÇÃO DA MATRIZ QUADRADA

for line in range(mtx_line):
    new_line = []
    
    for column in range(mtx_column):
        new_line.append(int(input(f"{line+1}m X {column+1}n: ")))
        
    matrix.append(new_line)
    
    

# CÁLCULO DO Sn

def SnFunc(mtx_size):
    i = 1
    Sn = 1
    
    while i <= mtx_size:
        Sn *= i
        i += 1
    
    return Sn

Sn = SnFunc(mtx_size)



# SIGMA (σ)

def sigmaFunc(mtx_size):
    sigma = []
    sgn = ''
    i = 1
    
    while i <= mtx_size:
        sgn += str(i)
        i += 1
    
    # PERMUTAÇÃO
    
    def permutation(lst):
        if len(lst) < 2:
            return [lst]
        
        l = []
        
        for i in range(len(lst)):
           m = lst[i]
           remLst = lst[:i] + lst[i+1:]
           
           for p in permutation(remLst):
               l.append([m] + p)
               
        return l
    
    
    for i in permutation([i+1 for i in range(mtx_size)]):
        sigma.append("".join(map(str,i)))
    
    return sigma
    
sigma = sigmaFunc(mtx_size)


# RESULTADO DA PERMUTAÇÃO
    
def permResult(sigma):
    base = sigma[0]   
    sigma_changes = []
        
    for i in range(len(sigma)):
        count = 0
        diff = sigma[i]
            
        for j in range(len(base)):
            count += diff[j] != base[j]
            
        resp = count -1 if count > 0 else 0
        sigma_changes.append(resp)
        
    return sigma_changes

sign_sigma = permResult(sigmaFunc(mtx_size))

    
    
# PRODUTÓRIO


def product(sigma,Sn):
    prod = []
    
    for j in range(len(sigma)):
        base = sigma[j]
    
        for i in range(len(base)):
            u = base[i]
            c = 1
            
            while u != str(c):
                c += 1
              
            a = matrix[i][c-1]
            e = ((-1)**sign_sigma[i])*a            
            
            prod.append(e)

    lis = []
    
    for i in range(Sn):
        li = []
        
        for j in range(mtx_size):
            li.append(prod[mtx_size*i+j])
            
        lis.append(li)
    
    
    
    prodResult = []
    
    for i in range(len(lis)):
        obj = lis[i]
        mult = 1
        
        for j in obj:
            mult *= j
        
        prodResult.append(mult)
    
    
        
    
    return prodResult


prod = product(sigma,Sn)

        
   
# SOMATÓRIA 


def summation(prod, sign_sigma):
    a = 0
    b = 0
    
    count = 0
    
    for i in prod:    
        if sign_sigma[count]%2 == 0:
            a += i
            count += 1
        
        else:
            b += i   
            count += 1
            
    result = a - b
    
    return result


soma = summation(prod, sign_sigma)
        
        
    
# CONCLUSÃO

if soma == 0:
    detA = "A matriz não contém inversa."

else:
    detA = "A matriz contém inversa."


        
print("\n---------ESTRUTURA DA MATRIZ----------\n")
for line in range(mtx_line):
    for column in range(mtx_column):
        print("[",matrix[line][column], "]",end="")

    print()


print("\n-----------RESULTADO FINAL------------\n")
print("Sn =",Sn)
print("sgn(σ) =",sigma)
print("∏ =",prod)
print("∑ =",soma)
print("\ndet(A) =",soma,"->",detA)