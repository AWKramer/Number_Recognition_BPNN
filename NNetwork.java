import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class NNetwork {
	
	public int numInputDimensions; 
	public int numInputNodes; 
	public int numHiddenNodes; 
	public int numOutputNodes; 
	
	public double[][] weightsInputHidden; //store the weights from input nodes to hidden nodes 
	public double[][] weightsHiddenOutput; //store the weights from hidden nodes to output nodes 
	
	public double[][] previousWeightChangeIH; //to store deltas at time t so we can use them with momentum in update at t+1 
	public double[][] previousWeightChangeHO; 
	
	public double[] hiddenLayerInputs; //store the net input to each hidden node
	public double[] outputLayerInputs; //store net input to each output node 
	
	public double[] inputLayerOutputs; //store output 
	public double[] hiddenLayerOutputs; 
	public double[] outputLayerOutputs; 
	
	public double[] outputLayerDeltas; 
	public double[] hiddenLayerDeltas; 
	
	public double learningRate; 
	public double momentum; 
	public int maxEpochs; 
	
	public double[][] currentTrainingOutputs; 
	
	public int numTrainingPatterns; 
	public int numTestPatterns; 
	
	public double[][] trainingInputs; 
	public double[][] targetTrainingOutputs; 
	
	public double[][] testInputs; 
	public double[][] testOutputs; 
	
	public static void main(String[] args) throws IOException {
		NNetwork nn = new NNetwork(64, 50, 10, 0.15, 0.8, 6000, 3823); 
	
		//Training phase
		nn.readTrainingData();
		nn.train(); 
		
		//Testing phase 
		nn.readTestData(); 
		int correct = nn.test();

		//Report network accuracy 
		System.out.println("Classification accuracy (test): " + (correct*100.0/nn.testInputs.length) + "%\nCorrect: " + correct + "\nIncorrect: " + (nn.testInputs.length - correct));  
		
	}

	//Initialise values in constructor 
	public NNetwork(int inputDimensions, int hiddenNodes, int outputNodes, double lr, double momentum, int maxEpochs, int numTrainingInputs) {
		this.numInputDimensions = inputDimensions; 
		this.numInputNodes = this.numInputDimensions+1; //for bias node   
		this.numHiddenNodes = hiddenNodes; 
		this.numOutputNodes = outputNodes; 
		this.learningRate = lr; 
		this.maxEpochs = maxEpochs; 
		this.weightsInputHidden = new double[numInputNodes][numHiddenNodes]; //accounts for the added bias weight to each hidden node 
		this.weightsHiddenOutput = new double[numHiddenNodes+1][numOutputNodes]; //to account for the added bias weight to each output node 
		this.hiddenLayerInputs = new double[numHiddenNodes]; 
		this.outputLayerInputs = new double[numOutputNodes]; 
		this.inputLayerOutputs = new double[numInputNodes]; 
		this.hiddenLayerOutputs = new double[numHiddenNodes]; 
		this.outputLayerOutputs = new double[numOutputNodes]; 
		this.outputLayerDeltas = new double[numOutputNodes]; 
		this.hiddenLayerDeltas = new double[numHiddenNodes]; 
		this.previousWeightChangeIH = new double[numInputNodes][numHiddenNodes]; 
		this.previousWeightChangeHO = new double[numHiddenNodes][numOutputNodes]; 
		this.numTrainingPatterns = numTrainingInputs; 
		this.trainingInputs = new double[numTrainingPatterns][numInputNodes]; 
		this.targetTrainingOutputs = new double[numTrainingPatterns][numOutputNodes]; 
		this.currentTrainingOutputs = new double[numTrainingPatterns][numOutputNodes]; 
		initialiseWeights(); 
		System.out.println("Network created!"); 
	}
	
	//Read in training data from file 
	public void readTrainingData() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader("training.txt")); 
		ArrayList<String> dataPoints = new ArrayList<String>(); 
		
		String s; 
		while ((s = br.readLine()) != null) { //read in data line by line 
			dataPoints.add(s); 
		}
		br.close(); 
		
		for (int i = 0; i < this.numTrainingPatterns; i++) {
			String[] points = dataPoints.get(i).split(","); 
			if (points.length != 65) {
				System.out.println("WARNING: Data point does not have 65 dimensions!");
			}
			for (int j = 0; j < points.length-2; j++) {
				this.trainingInputs[i][j] = Integer.parseInt(points[j]); 
			}
			
			this.trainingInputs[i][points.length-1] = 1; //bias entry 
			
			for (int k = 0; k < this.numOutputNodes; k++) {
				int outputClass = Integer.parseInt(points[points.length-1]); 
				if (k==outputClass) {
					this.targetTrainingOutputs[i][k] = 0.9; //since we're using a sigmoid function we need approximate values  
				} else {
					this.targetTrainingOutputs[i][k] = 0.1; 
				}
			}
		}
	}
	
	//Read in test data from file 
	public void readTestData() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader("testing.txt")); 
		ArrayList<String> dataPoints = new ArrayList<String>(); 
		
		String s; 
		while ((s = br.readLine()) != null) { //read in data line by line 
			dataPoints.add(s); 
		}
		br.close(); 
		
		this.testInputs = new double[dataPoints.size()][numInputNodes];
		this.testOutputs = new double[dataPoints.size()][numOutputNodes]; 
		
		for (int i = 0; i < dataPoints.size(); i++) {
			String[] points = dataPoints.get(i).split(","); 
			if (points.length != 65) {
				System.out.println("WARNING: Data point does not have 65 dimensions!");
			}
			for (int j = 0; j < points.length-2; j++) {
				this.testInputs[i][j] = Integer.parseInt(points[j]); 
			}
			
			this.testInputs[i][points.length-1] = 1; //bias entry 
			
			for (int k = 0; k < this.numOutputNodes; k++) {
				int outputClass = Integer.parseInt(points[points.length-1]); 
				if (k==outputClass) {
					this.testOutputs[i][k] = 0.9; //since we're using a sigmoid function we need approximate values  
				} else {
					this.testOutputs[i][k] = 0.1; 
				}
			}
		}
	}
	
	//Initialise weights to random values bounded between 0.5 and -0.5 
	public void initialiseWeights() {
		//initialise input to hidden layer weights
		for (int i = 0; i < this.numInputNodes; i++) {
			for (int j = 0; j < this.numHiddenNodes; j++) {
				this.weightsInputHidden[i][j] = Math.random() - 0.5; //return values between -0.5 and 0.5 
//				System.out.println("Weight from input node " + i + " to hidden node " + j + ": " + this.weightsInputHidden[i][j]); 
			}
		}
		
		//initialise hidden to output layer weights
		for (int i = 0; i < this.numHiddenNodes; i++) {
			for (int j = 0; j < this.numOutputNodes; j++) {
				this.weightsHiddenOutput[i][j] = Math.random() - 0.5; //return values between -0.5 and 0.5 
//				System.out.println("Weight from hidden node " + i + " to output node " + j + ": " + this.weightsHiddenOutput[i][j]); 
			}
		}
		
		System.out.println("Weights initialised."); 
	}
	
	//Training method 
	public void train() {
		double mse = Integer.MAX_VALUE;  
		int epoch = 0; 
		double[] pattern; 
		double[] desiredOutput; 
		
		do {
			epoch++; 
			
			for (int i = 0; i < this.numTrainingPatterns; i++) {
				pattern = this.trainingInputs[i]; 
				desiredOutput = this.targetTrainingOutputs[i]; 
				this.currentTrainingOutputs[i] = feedforward(pattern); 
				backpropagate(pattern, desiredOutput, this.currentTrainingOutputs[i]); 
//				printNetworkState(); 
				
			}
			
			if (epoch % 100 == 0) {
				System.out.println("Epoch: " + epoch); 
				mse = computeMSE(); 
				System.out.println("MSE: " + mse);
			}

//			printNetworkState(); 
			
		} while (mse > 0.0008 && epoch < this.maxEpochs); 
		
		System.out.println("Network trained. Calculating classification accuracy..."); 
		
		//calculate # correct 
		int correct = calculateClassificationAccuracy(); 
		
		System.out.println("Classification accuracy (training): " + (correct*100.0/this.numTrainingPatterns) + "%\nCorrect: " + correct + "\nIncorrect: " + (this.numTrainingPatterns - correct)); 
	}
	
	public double[] feedforward(double[] pattern) {
		
		//assign outputs of input layer
		for (int i = 0; i < this.numInputNodes; i++) {
			this.inputLayerOutputs[i] = pattern[i]; 
		}
		
		//assign inputs of hidden layer
		for (int i = 0; i < this.numHiddenNodes; i++) {
			double sum = 0; 
			for (int j = 0; j < this.numInputNodes; j++) {
				sum += this.weightsInputHidden[j][i] * this.inputLayerOutputs[j]; 
			}
			this.hiddenLayerInputs[i] = sum; 
		}
		
		//work out outputs of hidden layer
		for (int i = 0; i < this.numHiddenNodes; i++) {
			 this.hiddenLayerOutputs[i] = sigmoid(this.hiddenLayerInputs[i]); 
		}		
		
		//assign inputs of output layer 
		for (int i = 0; i < this.numOutputNodes; i++) {
			double sum = 0; 
			for (int j = 0; j < this.numHiddenNodes; j++) {
				sum += this.weightsHiddenOutput[j][i] * this.hiddenLayerOutputs[j];  
			}
			this.outputLayerInputs[i] = sum; 
		}
		
		//work out output of output layer 
		for (int i = 0; i < this.numOutputNodes; i++) {
			 this.outputLayerOutputs[i] = sigmoid(this.outputLayerInputs[i]); 
		}
		
		return this.outputLayerOutputs; 
	}
	
	public double sigmoid(double input) {
		return 1 / (1 + Math.pow(Math.E, -input)); 
	}
	
	public void backpropagate(double[] pattern, double[] targetOutput, double[] actualOutput) {
		double error; 
		double sum; 
		
		//work out the error and deltas of each output node
		for (int i = 0; i < this.numOutputNodes; i++) {
			if (targetOutput[i] == 0.9 && actualOutput[i] >= 0.9) {
				error = 0; 
			} else if (targetOutput[i] == 0.1 && actualOutput[i] <= 0.1) {
				error = 0;  
			} else {
				error = targetOutput[i] - actualOutput[i];
			}
			 
			this.outputLayerDeltas[i] = error * actualOutput[i] * (1-actualOutput[i]); 
		}
		
		//adjust the weights from hidden layer to output layer using momentum 
		double change; 		
		for (int out = 0; out < this.numOutputNodes; out++) {
			for (int hid = 0; hid < this.numHiddenNodes; hid++) {
				change = (this.learningRate * this.hiddenLayerOutputs[hid] * this.outputLayerDeltas[out]) + (this.momentum * this.previousWeightChangeHO[hid][out]); 
				this.weightsHiddenOutput[hid][out] += change; 
				this.previousWeightChangeHO[hid][out] = change; 
			}
			this.weightsHiddenOutput[this.numHiddenNodes][out] += this.learningRate * this.outputLayerDeltas[out]; //update bias 
		}
		
		//work out deltas of hidden layer nodes 
		for (int i = 0; i < this.numHiddenNodes; i++) {
			sum = 0; 
			for (int j = 0; j < this.numOutputNodes; j++) {
				sum += this.outputLayerDeltas[j] * this.weightsHiddenOutput[i][j]; 
			}
			error = sum * this.hiddenLayerOutputs[i] * (1 - this.hiddenLayerOutputs[i]); 
			this.hiddenLayerDeltas[i] = error; 
		}
		
		//adjust weights from input to hidden layer 
		for (int hid = 0; hid < this.numHiddenNodes; hid++) {
			for (int in = 0; in < this.numInputNodes; in++) {
				change = (this.learningRate * this.hiddenLayerDeltas[hid] * this.inputLayerOutputs[in]) + (this.momentum * this.previousWeightChangeIH[in][hid]); 
				this.weightsInputHidden[in][hid] += change; 
				this.previousWeightChangeIH[in][hid] = change; 
			}
			this.weightsInputHidden[this.numInputNodes-1][hid] += this.learningRate * this.hiddenLayerDeltas[hid]; //update bias 
		}
		
	}
	
	public double computeMSE() {
		double sum = 0; 
		for (int i = 0; i < this.numTrainingPatterns; i++) {
			double[] output = feedforward(this.trainingInputs[i]); 
			for (int j = 0; j < output.length; j++) {
				if (this.targetTrainingOutputs[i][j] == 0.9 && output[j] >= 0.9) {
					sum += 0; 
				} else if (this.targetTrainingOutputs[i][j] == 0.1 && output[j] <= 0.1) {
					sum += 0; 
				} else {
					sum += Math.pow((this.targetTrainingOutputs[i][j]-output[j]), 2);  
				}
			}
		}
		return sum / this.numTrainingPatterns; 
	}
	
	public int calculateClassificationAccuracy() {
		int correct = 0; 
		for (int i = 0; i < this.numTrainingPatterns; i++) {
			
			double[] output = feedforward(this.trainingInputs[i]); 
			double[] target = this.targetTrainingOutputs[i]; 
			int outputMax = findMaxIndex(output);
			int targetMax = findMaxIndex(target); 
			boolean allCorrect = true; 
			
			for (int j = 0; j < this.numOutputNodes; j++) {
				if (target[j] == 0.9 && output[j] >= 0.9) {
					//correct 
				} else if (target[j] == 0.1 && output[j] <= 0.1) {
					//correct
				} else {
					allCorrect = false; 
				}
			}
			
			if (allCorrect) {
				correct++; 
			}
			
			System.out.println("Training pattern " + (i+1) + ": Target class - " + targetMax + " Network class - " + outputMax); 

		}
		return correct; 
	}
	

	//Test method 
	public int test() {
		int correct = 0; 
		for (int i = 0; i < this.testInputs.length; i++) {
			double[] output = feedforward(this.testInputs[i]); 
			double[] target = this.testOutputs[i]; 
			int outputMax = findMaxIndex(output); 
			int targetMax = findMaxIndex(target); 
			if (output[outputMax] >= 0.9 && outputMax==targetMax) {
				correct++; 
			}
			System.out.println("Test pattern " + (i+1) + ": Target class - " + targetMax + " Predicted class - " + outputMax); 
		}
		return correct; 
	}
	
	private int findMaxIndex(double[] v) {
		int index = 0; 
		double max = 0; 
		for (int i = 0; i < v.length; i++) {
			if (v[i] > max) {
				max = v[i]; 
				index = i; 
			}
		}
		return index; 
	}

}
