import numpy as np
import pulp
import time
from scipy import stats

# funcion para calcular la suma de una lista
def suma(x):
    s=0
    for i in x:
        s=s+i
    return s

# funcion para el minimo de la lista X considerando solo los indices_buenos en la busqueda
def min_sin(X,indices_buenos,lim):
    mini=lim
    j=None
    for i in indices_buenos:
        if X[i]<mini:
            j=i     
            mini=X[j]
    return j

#funcion para encontrar calcular la penalizacion de una fila o columna
def pen(X,lim):
    indice=X.argmin() #primera busqueda lineal
    aux=X[indice]
    X[indice]=lim 
    indice2=X.argmin() #segunda busqueda lineal con el primer minimo cambiado
    X[indice]=aux #restaurar el valor del primer minimo
    valor = X[indice2]-X[indice]
    return valor

#funcion que recupera el indice del valor b en la lista C
def index(C,b):
    valor=None
    for i in range(len(C)):
        if C[i]==b:
            valor=i
            break
    return valor

# funcion que realiza cada iteración de del metodo de vogel
def vogel_it(C,costos,posiciones,n_s,m_s,maq,lim,P_fila,P_col,n_max,indices_fila,indices_col,fila,col):
    #C = matriz de costos
    #costos = vector de costos de las asignaciones
    #posiciones = vector de asignaciones
    #n_s = maquinas sin tener al menos una tarea asignada
    #m_n = tareas sin asignar
    #maq = vector que representa la cantidad de tareas asignadas por maquinas
    #lim = numero grande para tachar a las columnas y filas
    #P_fila = vector que guarda las penalizaciones de las filas validas (no tachadas) de la matriz C, cambia de tamaño en cada iteracion
    #P_col = vector que guarda las penalizaciones de las columnas validas (no tachadas) de la matriz C, cambia de tamaño en cada iteracion
    #n_max = numero maximo de tareas asignables a un sola maquina
    #indices_fila = vector que guarda los indices validos de las filas (no tachados) de la matriz C, del mismo tamaño que P_fila
    #indices_col = vector que guarda los indices validos de las columna (no tachados) de la matriz C, del mismo tamaño que P_col
    if col==True: #indicador si una columna fue tachada
        for i in range(len(P_fila)): # recorre el vector
            P_fila[i]=pen(C[indices_fila[i],indices_col],lim) #calcula la penalizacion en la fila valida i y todos los elementos dentro de las columnas validas
    if fila==True: #indicador si una fila fue tachada
        for i in range(len(P_col)): # recorre el vector
            P_col[i]=pen(C[indices_fila,indices_col[i]],lim)# calcula la penalización en la columna valida i y todos los elementos dentro de las filas validas
    col=False #el indicador inicia como falso
    fila=False #el indicador inicia como falso
    j=P_fila.argmax() #calcula el indice de la mayor penalizacion por fila 
    k=P_col.argmax() #calcula el indice de la mayor penalizacion por columna
    max_fila=P_fila[j] #la mayor penalizacion por fila
    max_col=P_col[k] #la mayor penalizacion por columna
    if max_fila>max_col: # busca la mayor penalizacion entre fila y columna
        #ind_fila y ind_col son los indices reales de la matriz C, que guarda la asignacion
        ind_fila=indices_fila[j] #si la pen. mayor por fila es mayor a la de columna, el indice se busca en el vector de indices
        ind_col=min_sin(C[ind_fila,:],indices_col,lim) #se busca el minimo de la fila ind_fila, con los indices buenos como los indices validos (indices_col)
        col_tachada=index(indices_col,ind_col) #la columna tachada es de la posicion de la asignacion, entonces se busca el indice real en el vector de indices
    else: #en caso contrario es similar a la fila
        ind_col=indices_col[k]
        ind_fila=min_sin(C[:,ind_col],indices_fila,lim)
        col_tachada=k #en este caso la indice de la columna es precisamente el mismo que el indice de P_col
    posicion=ind_fila+1,ind_col+1 # guarda la asigancion elegida (maquina, tarea)
    posiciones.append(posicion) #agrega la asigancion
    costos.append(C[ind_fila,ind_col]) # guarda el costo asociado a esa posicion
    C[:,ind_col]=lim # se tacha la columna de esa tarea al ser asignada
    P_col=np.delete(P_col,col_tachada) #se elimina la columna tachada de los vectores validos
    indices_col=np.delete(indices_col,col_tachada)
    col=True #como se tacho una columna, el indicar vuelve a True
    maq[ind_fila]+=1 # al vector de maquinas se le agrega una tarea a la maquina asignada
    n_s=maq.count(0) # n_s representa el numero de maquinas sin asignar, se cuentan las maquinas sin tareas, ya que se pueden asignar mas de una tarea a las maquinas 
    m_s=m_s-1 # las tareas sin asignar disminuyen en una unidad
    if maq[ind_fila]==n_max:  # si en la maquina recien asiganada alcanza el maximo de tareas, esa maquina de tacha
        C[ind_fila,:]=lim # se tacha la maquina
        fila_tachada=index(indices_fila,ind_fila) #la fila tachada es de la posicion de la asignacion, entonces se busca el indice real en el vector de indices
        P_fila=np.delete(P_fila,fila_tachada) # se elimina la fila tachada de los vetores validos
        indices_fila=np.delete(indices_fila,fila_tachada)
        fila=True #como se tacho una fila, el indicar vuelve a True
    if n_s==m_s: # si el numero de tareas sin asignar es igual al numero de maquinas sin asignar, todas las maquinas que tengan asignacion deben ser tachadas
        for i in indices_fila: 
            if maq[i]>0: #si la fila tiene una asignacion es tachada
                #se tacha la fila, igual al bloque anterior
                C[i,:]=lim 
                fila_tachada=index(indices_fila,i) 
                P_fila=np.delete(P_fila,fila_tachada)
                indices_fila=np.delete(indices_fila,fila_tachada)
                fila=True
    if m_s==1: # si exactamente queda una tarea sin asignar se le asigna directamente a la celda con menor costo en esa columna y nos ahorramos los pasos anteriores
        ind_col=indices_col[0] #en este momento indices_col es de largo 1, donde esta el unico indice valido
        ind_fila=min_sin(C[:,ind_col],indices_fila,lim) #se busca el minimo
        posicion=ind_fila+1,ind_col+1 #se guarda la posicion 
        posiciones.append(posicion) #se agrega al vector
        costos.append(C[ind_fila,ind_col]) # su guarda el costo
        C[:,ind_col]=lim # se tacha la columna de esa tarea al ser asignada
        col_tachada=0 #se tacha la columna
        P_col=np.delete(P_col,col_tachada)
        indices_col=np.delete(indices_col,col_tachada)
        col=True # se vuelve True por tachar
        maq[ind_fila]+=1 # al vector de maquinas se le agrega una tarea a la maquina asignada
        n_s=0 # se cuenta las maquinas sin asignar, siempre es 0
        m_s=m_s-1 # se cuentan las tareas sin asignar, siempre es 0
    return C,costos,posiciones,n_s,m_s,maq,indices_fila,indices_col,P_fila,P_col,fila,col

# funcion que resuelve el problema con el metodo de vogel y contiene la funcion de vogel de cada itaracion, ademas revuelve un resumen de los resultados del problema resuelto
def vogel(matriz): 
    C=np.array(matriz) # se trasnforma de una matris de lista a una de numpy
    pos=[] #vector que guarda las posiciones (asignaciones)
    costos=[] # vector que guarda los costos de las asignaciones
    n,m=C.shape # cantidad de maquinas y tareas
    n_s=n # maquinas = filas, n_s son las maquinas sin asignar
    m_s=m # tareas = columnas, m_s son las tareas sin asignar
    maq=[0 for i in range(n_s)] # vector que representa la cantidad de tareas asignadas por maquinas
    v_max=m_s-n_s+2 #numero maximo de tareas asignables por maquina
    lim=2*C.max() # valor que se usa para tachar la celda
    P_fila=np.array([0 for i in range(n)]) # vectores que guardan las penalizaciones de las filas y columnas validas (no tachadas) de la matriz C
    P_col=np.array([0 for i in range(m)]) #estos vectores van disminuyendo de tamaño a medida que pasan las iteraciones
    indices_fila=np.array([i for i in range(n)]) #vector que guarda los indices validos (no tachados) de la matriz C
    indices_col=np.array([i for i in range(m)]) # estos vectores son del mismo tamaño que los vectores de penalizaciones
    fila=True #indicador si una fila ha sido tachada
    col=True #indicador si una columna ha sido tachada
    resumen=[] # vector que guarda los principales resultados para imprimirlos al final
    inicio=time.time() # inicio del conteo del tiempo
    while m_s>0: # el metodo itera hasta que todas las tareas esten asignadas
        #una iteracion del metodo
        C,costos,pos,n_s,m_s,maq,indices_fila,indices_col,P_fila,P_col,fila,col=vogel_it(C,costos,pos,n_s,m_s,maq,lim,P_fila,P_col,v_max,indices_fila,indices_col,fila,col)
    fin=time.time() # fin del conteo del tiempo
    resumen.append(suma(costos)) # suma total de costos
    resumen.append(round(fin-inicio,6)) # guarda el tiempo de solucion
    resumen.append(pos) # asignaciones realizadas
    return resumen

# funcion de lectura de datos del problema donde los costos estan separados por un separador, los guarda en una lista
def leer(directorio,separador):
    f=open(directorio,'r') #se abre en modo lectura
    C=[]
    for linea in f.readlines():
        linea_v=linea.split(separador)
        for j in range(len(linea_v)):
            if j==len(linea_v)-1:
                linea_v[j]=linea_v[j].strip('\n')
            linea_v[j]=float(linea_v[j]) # se usan numeros continuos
        C.append(linea_v)
    f.close()
    return C

# funcion que resuelve el problema con un metodo exacto y contiene la funcion de vogel de cada itaracion, ademas revuelve un resumen de los resultados del problema resuelto
def uap_pulp(matriz,nombre):
    n=len(matriz) # filas = maquinas
    m=len(matriz[0]) # columnas = tareas
    resumen=[] # vector que guarda los principales resultados para imprimirlos al final
    # DEFINICIÓN DEL PROBLEMA
    # PARAMETROS
    maquinas=[i for i in range(1,n+1)] # enumeracion de maquinas
    tareas=[i for i in range(1,m+1)] # enumeracion de tareas
    arcos=[(i,j) for i in maquinas for j in tareas] # enumeracion de todas las combinaciones posibles entre maquinas y tareas
    # VARIABLES
    X = pulp.LpVariable.dicts("X",arcos,cat='Continuous',lowBound=0) # diccionario de variables
    # FUNCION OBJETIVO
    mdl = pulp.LpProblem(nombre, pulp.LpMinimize) # modelo de minimizacion
    mdl += pulp.lpSum([X[(i,j)]*matriz[i-1][j-1] for i in maquinas for j in tareas]) #funcion objetivo
    # RESTRICCIONES
    for i in maquinas:
        mdl += pulp.lpSum(X[(i,j)] for j in tareas)>=1 # las tareas asignables a una maquina es minima 1
    for j in tareas:
        mdl += pulp.lpSum(X[(i,j)] for i in maquinas)==1 # todas las tareas deben ser asignadas
    # SOLUCIÓN
    solver = pulp.getSolver('CPLEX_CMD',threads=1) #asigancion del solver de CPLEX, threads=1 -> forma secuencial
    inicio=time.time() # inicio del conteo del tiempo
    status = mdl.solve(solver) # resolver el problema
    fin=time.time() # fin del conteo del tiempo
    costos=[] # vector de costos 
    pos=[] # vector de posiciones (asignaciones)
    for v in mdl.variables(): # si la variables es 1 se obtienen las posiciones y el costo
        if v.varValue ==1: #guarda la informacion si la variable es 1
            # print(v,'=',v.varValue)
            i,j=indices(v) #obtiene los indices desde la variables
            costos.append(matriz[i-1][j-1])
            pos.append((i,j))
    if 'infeasible' not in pulp.LpStatus[status]: # si el problema no es infactible guarda la informacion resumen
        resumen.append(suma(costos)) # guarda los costos
        resumen.append(round(fin-inicio,6)) # guarda el tiempo de solucion
        resumen.append(pos) # guarda las posiciones
    else:
        print('Infactible')
    return resumen

# funcion que obtiene las posiciones de las varibles
def indices(v):
    name=v.name
    name=name.strip('X_(')
    name=name.strip(')')
    [i,j]=name.split(',_')
    i=int(i)
    j=int(j)
    return i,j

#direccion de la capeta donde esta todo
carpeta="C:\\Users\\usuario\\Documents\\UNIVERSIDAD\\SEXTO\\MEMORIA\\codigo\\" 

##Instancia de la Literatura
#nombres_literatura.txt es un archivo con los nombres de los problemas de la literatura
f=open(carpeta+"nombres_literatura.txt")
r_lit=open(carpeta+"Resultados_Literatura.txt",'w') #escribir los resultados
direc_lit=carpeta+'literatura\\' #carpeta donde estan las instancias de la literatura
for linea in f.readlines():
    linea=linea.strip('\n') #linea es cada nombre del problema
    matriz=leer(direc_lit+linea,',') # lectura del problema y se obtiene la matriz de costos
    nombre=linea.strip('.txt') #nombre del problema
    c_v,t_v,p_v=vogel(matriz) #solucion del problema con el metodod de vogel
    c_p,t_p,p_p=uap_pulp(matriz,nombre) #solucion del problema con el metodo exacto
    r_lit.write(nombre+'\n')
    r_lit.write('            |   PULP    |   VOGEL   |\n')
    r_lit.write('     Costo: | '+'{0:9}'.format(c_p)+' | '+'{0:9}'.format(c_v)+' |\n')
    r_lit.write('    Tiempo: | '+'{0:9.2}'.format(t_p)+' | '+'{0:9.2}'.format(t_v)+' |\n')
    r_lit.write('Asig. PULP:'+str(p_p)+'\n')
    r_lit.write('Asig.VOGEL:'+str(p_v)+'\n')
    r_lit.write('----------------------------------------------------------------------\n')
f.close()
r_lit.close()

#200 Instancias
nombres_200=carpeta+"nombres_200_todos.txt" #el nombre de los 200 problemas
f=open(nombres_200,'r')
costo_vogel=[]
costo_pulp=[]
tiempo_vogel=[]
tiempo_pulp=[]
r_200=open(carpeta+"Resultados_200.txt",'w') #escribir los resultados
r_200.write('Problema,Maquinas,Tareas,Costo PULP,Costo VOGEL,Tiempo PULP,Tiempo VOGEL\n')
direc_200=carpeta+"UAP200\\"
for linea in f.readlines():
    linea=linea.strip('\n') #linea es cada nombre del problema 
    matriz=leer(direc_200+linea,'\t') # lectura del problema y se obtiene la matriz de costos
    n=len(matriz)
    m=len(matriz[0])
    nombre=linea.split('_')[0]+'_('+str(n)+'x'+str(m)+')' #nombre del problema
    c_v,t_v,p_v=vogel(matriz) #solucion del problema con el metodo de vogel
    c_p,t_p,p_p=uap_pulp(matriz,nombre) #solucion del problema con el metodo exacto
    costo_vogel.append(c_v)
    tiempo_vogel.append(t_v)
    costo_pulp.append(c_p)
    tiempo_pulp.append(t_p)
    r_200.write(str(nombre)+','+str(n)+','+str(m)+','+str(c_p)+','+str(c_v)+','+str(t_p)+','+str(t_v)+'\n')
f.close()
r_200.close()

#Prueba de Hipotesis
costo_vogel=np.array(costo_vogel)
tiempo_vogel=np.array(tiempo_vogel)
costo_pulp=np.array(costo_pulp)
tiempo_pulp=np.array(tiempo_pulp)
r_h=open(carpeta+"Resultados_Pruebas_Hipotesis.txt",'w') #escribir los resultados

#Prueba de igualdad de Medias para los costos
n_v=200
m_v=np.mean(costo_vogel)
ds_v=np.std(costo_vogel)
n_p=200
m_p=np.mean(costo_pulp)
ds_p=np.std(costo_pulp)
Z=round(( m_p - m_v ) / pow((ds_p**2)/(n_p) + (ds_v**2)/(n_v),1/2),4)
Z_critico=round(stats.norm.ppf(0.05),4)
print('--------------------------------------------------------------')
print('Prueba de Hipotesis de igualdad de medias para los costos:')
print('Media Pulp (mu_1)=',m_p)
print('Media Vogel (mu_2)=',m_v)
print('Desviación Estandar Pulp (zigma_1)=',ds_p)
print('Desviación Estandar Vogel (zigma_2)=',ds_v)
print('H_0= mu_1 = m_2')
print('H_A= mu_1 < m_2')
print('Z calculado=',Z)
print('Z critico (95%)=',Z_critico)
r_h.write('--------------------------------------------------------------\n')
r_h.write('Prueba de Hipotesis de igualdad de medias para los costos:\n')
r_h.write('Media Pulp (mu_1)= '+str(m_p)+'\n')
r_h.write('Media Vogel (mu_2)= '+str(m_v)+'\n')
r_h.write('Desviación Estandar Pulp (zigma_1)= '+str(ds_p)+'\n')
r_h.write('Desviación Estandar Vogel (zigma_2)= '+str(ds_v)+'\n')
r_h.write('H_0= mu_1 = m_2\n')
r_h.write('H_A= mu_1 < m_2\n')
r_h.write('Z calculado='+str(Z)+'\n')
r_h.write('Z critico (95%)='+str(Z_critico)+'\n')
if Z>Z_critico:
    r_h.write('Se aceptó la H_0, las medias de los costos son significativamentes Iguales\n')
    print('Se aceptó la H_0, Las medias de los costos son significativamentes Iguales')
else:
    r_h.write('Se rechazó la H_0, Las medias de los costos son significativamentes Distintas\n')
    print('Se rechazó la H_0, Las medias de los costos son significativamentes Distintas')

#Prueba de igualdad de Medias para los tiempos computacionales
n_v=200
m_v=np.mean(tiempo_vogel)
ds_v=np.std(tiempo_vogel)
n_p=200
m_p=np.mean(tiempo_pulp)
ds_p=np.std(tiempo_pulp)
Z=round(( m_p - m_v ) / pow((ds_p**2)/(n_p) + (ds_v**2)/(n_v),1/2),4)
Z_critico=round(stats.norm.ppf(0.95),4)
print('--------------------------------------------------------------')
print('Prueba de Hipotesis de igualdad de medias para los tiempos:')
print('Media Pulp (mu_1)=',m_p)
print('Media Vogel (mu_2)=',m_v)
print('Desviación Estandar Pulp (zigma_1)=',ds_p)
print('Desviación Estandar Vogel (zigma_2)=',ds_v)
print('H_0= mu_1 = m_2')
print('H_A= mu_1 > m_2')
print('Z calculado=',Z)
print('Z critico (95%)=',Z_critico)
r_h.write('--------------------------------------------------------------\n')
r_h.write('Prueba de Hipotesis de igualdad de medias para los tiempos:\n')
r_h.write('Media Pulp (mu_1)= '+str(m_p)+'\n')
r_h.write('Media Vogel (mu_2)= '+str(m_v)+'\n')
r_h.write('Desviación Estandar Pulp (zigma_1)= '+str(ds_p)+'\n')
r_h.write('Desviación Estandar Vogel (zigma_2)= '+str(ds_v)+'\n')
r_h.write('H_0= mu_1 = m_2\n')
r_h.write('H_A= mu_1 > m_2\n')
r_h.write('Z calculado= '+str(Z)+'\n')
r_h.write('Z critico (95%)= '+str(Z_critico)+'\n')
if Z<Z_critico:
    r_h.write('Se aceptó la H_0, las medias de los tiempos son significativamentes Iguales\n')
    print('Se aceptó la H_0, Las medias de los tiempos son significativamentes Iguales')
else:
    r_h.write('Se rechazó la H_0, Las medias de los tiempos son significativamentes Distintas\n')
    print('Se rechazó la H_0, Las medias de los tiempos son significativamentes Distintas')
r_h.close()
print('Terminado.')