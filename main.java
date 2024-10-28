import java.util.*;
import java.io.Serializable;


class Neurona implements Serializable {
	private static final long serialVersionUID = 8729854615844306332L;

	public ArrayList<Sinapsis> entradas, salidas;
	private double resultado;

	private double gradiente;
	
	private double p_umbral;
	private double delta_p_umbral = 0.0;

	public Neurona(boolean conUmbral) {
		if(conUmbral) {
			p_umbral=Math.random();
		}
		this.entradas = new ArrayList<Sinapsis>();
		this.salidas = new ArrayList<Sinapsis>();
	}

	public double salidaNeurona() {
		double suma = p_umbral;
		for(int i=0; i<entradas.size();i++) {
			suma += entradas.get(i).inicio.resultado * entradas.get(i).peso;
		}
		resultado = Math.tanh(suma);
		return resultado;
	}

	public double salidaNeuronaDerivada() {
		return 1-(resultado*resultado);
	}

	public void calculaGradientesNeuronaOculta(){
		double suma = 0.0;
		for(int i=0; i<salidas.size(); i++) {
			suma+=salidas.get(i).peso*salidas.get(i).fin.gradiente;
		}
		gradiente = suma*salidaNeuronaDerivada();
	}

	public void calculaGradientesNeuronaSalida(double valorEsperado) {
		gradiente = (valorEsperado-resultado) * salidaNeuronaDerivada();
	}

	public void actualizarPesos(double razonAprendizaje, double momento) {
		for(int i=0; i<entradas.size(); i++) {
			double deltaPesoViejo = entradas.get(i).deltaWeight;
			double deltaPesoNuevo = razonAprendizaje * entradas.get(i).inicio.resultado * gradiente + momento * deltaPesoViejo;
			entradas.get(i).deltaWeight = deltaPesoNuevo;
			entradas.get(i).peso += deltaPesoNuevo;
		}
		double deltaPesoViejo = delta_p_umbral;
		double deltaPesoNuevo = razonAprendizaje * resultado * gradiente + momento * deltaPesoViejo;

		delta_p_umbral = deltaPesoNuevo;
		p_umbral += deltaPesoNuevo;
	}

	public double getResultado() {
		return resultado;
	}
	
	public void setResultado(double resultado) {
		this.resultado = resultado;
	}
	
	public double getGradiente() {
		return gradiente;
	}
	
	public ArrayList<Sinapsis> getEntradas(){
		return entradas;
	}
	
	public ArrayList<Sinapsis> getSalidas(){
		return salidas;
	}
//Fin clase Neurona
}

class Red implements Serializable {
	private static final long serialVersionUID = 3286199902474469057L;
	private double RAZON_APRENDIZAJE;
	private double MOMENTO;
	private ArrayList<Neurona[]> network;

	public Red(int inputs, int[] capas, int outputs, double razon_Aprendizaje, double momento)
		throws RuntimeException{
			if(inputs <1 || outputs<1 || capas.length<1 || razon_Aprendizaje <=0 ||momento<=0) {
				throw new RuntimeException();
			}
			this.RAZON_APRENDIZAJE = razon_Aprendizaje;
			this.MOMENTO = momento;
			this.network = new ArrayList<Neurona[]>(inputs + capas.length + outputs);
			//Inicializacion de las neuronas
			Neurona[] capa_temporal = new Neurona[inputs];
			for(int i=0; i<inputs; i++) {
				capa_temporal[i] = new Neurona(false); //Sin umbral (es de entrada)
			}
			network.add(capa_temporal);
			Neurona[] array;
			for(int i=0; i<capas.length; i++) {
				array = new Neurona[capas[i]];
				for(int j=0;j<array.length;j++) {
					array[j] = new Neurona(true); //Con umbral
				}
				network.add(array); //Añadir la capa oculta creada
			}
			capa_temporal = new Neurona[outputs];
			for(int i=0; i<outputs; i++) {
				capa_temporal[i] = new Neurona(true);
			}
			network.add(capa_temporal);
			for(int i=1; i<network.size(); i++) {
				for(int j=0; j<network.get(i).length; j++) {
					for(int k=0; k<network.get(i-1).length; k++) {
						Sinapsis s = new Sinapsis(network.get(i-1)[k], network.get(i)[j], Math.random());
						network.get(i)[j].entradas.add(s);
						network.get(i-1)[k].salidas.add(s);
					}
				}
			}
		}
	
	public double[] epoca(ArrayList<Double> inputs) {
		double[] outs = new double[network.get(network.size()-1).length];
		if(inputs.size()!=network.get(0).length) {
			return null;
		}
		for(int i=0; i<inputs.size();i++) {
			network.get(0)[i].setResultado(inputs.get(i));
		}
		for(int i=1; i<network.size();i++) {
			for(int j=0; j<network.get(i).length; j++) {
				network.get(i)[j].salidaNeurona();
			}
		}
		for(int i=0; i<network.get(network.size()-1).length;i++) {
			outs[i] = network.get(network.size()-1)[i].getResultado();
		}
		return outs;
	}

	public void calibrar(double[] valoresObjetivo) {
		if(valoresObjetivo.length != network.get(network.size()-1).length)
			return;
		
		//Calcular el gradiente en las neuronas de salida
		for(int i=0; i<network.get(network.size()-1).length;i++) {
			network.get(network.size()-1)[i].calculaGradientesNeuronaSalida(valoresObjetivo[i]);
		}
		//Calcular el gradiente de las neuronas en las capas ocultas
		for(int i = network.size()-2; i>0; i--) {
			for(int j = 0; j<network.get(i).length; j++){
				network.get(i)[j].calculaGradientesNeuronaOculta();
			}
		}

		//neurona actualiza sus sinapsis de entrada)
		for(int i=network.size()-1; i>0; i--) {
			for(int j=0; j<network.get(i).length;j++) {
				network.get(i)[j].actualizarPesos(RAZON_APRENDIZAJE, MOMENTO);
			}
		}
	}
	public double getMOMENTO() {return MOMENTO;}
	public double getRAZON_APRENDIZAJE() {return RAZON_APRENDIZAJE;}

// Fin clase Red
}

class Sinapsis implements Serializable{
	private static final long serialVersionUID = 7597241645279827278L;
	public Neurona inicio, fin; //Neuronas de inicio y fin
	public double peso; //Peso asociado a la conexion
	public double deltaWeight = 0.0; //Variacion del peso con respecto al de la epoca anterior
	
	public Sinapsis(Neurona inicio, Neurona fin, double peso) {
		this.inicio = inicio;
		this.fin = fin;
		this.peso = peso;
		}
}


public class main{

public static void main(String[] args) {
	int errores=0;
	int intentos=0;
	double MOMENTO = 0.5;
	double RAZON_APRENDIZAJE = 0.2;
	int inputs = 2;
	int[] array_ocultas = {4};
	int outputs = 1;
	
	
	Red red = new Red(inputs,array_ocultas,outputs,RAZON_APRENDIZAJE,MOMENTO);
	
	Scanner scanner = new Scanner(System.in);
	System.out.println("Elige una opción: \n[1].- XOR \n[2].- NAND \n[3].- NOR \n[1-3]: ");
	int opcion = scanner.nextInt();

	double entradas[][] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double[] salidas = {0.0,0.0,0.0,0.0}; // Cambia aquí para usar un array

	switch(opcion) {
		case 1:
			// Tabla XOR
			salidas = new double[]{0.0, 1.0, 1.0, 0.0}; // Salidas esperadas
			System.out.println("Simulando entrenamiento para XOR:");
			break;
		case 2:
			// NAND
			salidas = new double[]{1.0, 1.0, 1.0, 0.0};
			System.out.println("Simulando entrenamiento para NAND:");
			break;
		case 3:
			// NOR
			salidas = new double[]{1.0, 0.0, 0.0, 0.0};
			System.out.println("Simulando entrenamiento para NOR:");
			break;
		default:
			System.out.println("Opción no válida.");
	}
	// Simular entrenamiento
	for (int etapa = 0; etapa < 8000; etapa++) {
		for (int i = 0; i < 4; i++) {
			ArrayList<Double> ins = new ArrayList<Double>(); // Declarar ins
			for (int j = 0; j < 2; j++) {
				ins.add(entradas[i][j]); // Rellenar la lista con las entradas
			}
			double[] arr = red.epoca(ins); // Valores retornados
			for (int j = 0; j < arr.length; j++) {
				double error = salidas[i] - arr[j];
				
				if(error <= 0.1){ //Para obtener salidas donde sea menor posible el error.
					System.out.println("Ciclo " + etapa + " Epoca: " + i + " Entrada:" + entradas[i][0] + " " + entradas[i][1] + " Salida :" + arr[j] + " Esperado: " + salidas[i] + " ERROR: " + error);
				}
				
				if (error >= 0.1)//Incrementa si es considerable el error.
					errores++;
				
				intentos++;
				double[] salidaEsperada = new double[1];
				salidaEsperada[0] = salidas[i];
				red.calibrar(salidaEsperada);
			}
		}
	}

	System.out.println("\n\nERRORES TOTALES:"+errores+",intentos:"+intentos*entradas.length );
//Fin clase Main
}
}