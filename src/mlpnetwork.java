/*
 * Author: Sebastin Justeeson
 * Description: Neural network program to solve the parity problem
 */




import java.util.Random;
import java.util.Scanner;

public class mlpnetwork {
	private double ABSOLUTE_ERROR = 0.05;
	private Boolean momentumMode = false;
	private double CURRENT_ERROR = Double.MAX_VALUE;
	private double inputAndHiddenLayerWeights[][];
	private double hiddenAndOutputLayerWeights[][];
	private double inputAndHiddenLayerWeightsDifference[][];
	private double hiddenAndOutputLayerWeightsDifference[][];
	private double hiddenLayerBiasDifference[];
	private double hiddenLayerBias[];
	private double outputLayerBias;
	private double outputLayerBiasDifference;
	private double learningRate = 0.05;
	private double learningMomentumAmount = 0.9;
	private double randMinimum = -1.0;
	private double randMaximum = 1.0;
	private double inputArray_1[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
	private double inputArray_2[] = {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0};
	private double inputArray_3[] = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0};
	private double inputArray_4[] = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
	private double inputArray[];
	private double desiredOutputArray[] = {0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0};
	private double hiddenLayerOutputArray[];
	private double obtainedOutput;
	private double outputLayerBiasOutput;
	private double sigmoidOutputArray[];
	private double sigmoidHiddenLayerArray[];
	private double deltaK;
	private double deltaJ;
	private int epoch = 0;
	
	
	public void initializeValues() {
		inputAndHiddenLayerWeights = new double[4][4];
		hiddenAndOutputLayerWeights = new double[4][1];
		inputAndHiddenLayerWeightsDifference = new double[4][4];
		hiddenAndOutputLayerWeightsDifference = new double[4][1];
		hiddenLayerBiasDifference = new double[4];
		
		hiddenLayerBias = new double[4];
		hiddenLayerOutputArray = new double[4];
		sigmoidOutputArray = new double[16];
		sigmoidHiddenLayerArray = new double[4];
		inputArray = new double[16];
		
		Random randomNum = new Random();
		
		// Initialize weights for input-hidden layer
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				double num = randomNum.nextDouble() * (randMaximum - randMinimum) + randMinimum;
				inputAndHiddenLayerWeights[i][j] = num;
			}
		}
			
		//Initialize the bias for hidden layer
		for(int j = 0; j < 4; j++) {
			double num = randomNum.nextDouble() * (randMaximum - randMinimum) + randMinimum;
			hiddenLayerBias[j] = num;
		}
		
		// Initialize weights for hidden-output layer
		for(int i = 0; i < 4; i++) {
			double num = randomNum.nextDouble() * (randMaximum - randMinimum) + randMinimum;
			hiddenAndOutputLayerWeights[i][0] = num;
		}
		
		// Initialize bias for output layer
		double num = randomNum.nextDouble() * (randMaximum - randMinimum) + randMinimum;
		outputLayerBias = num;
	}
	
	public void startLearning() {
		// Repeat for every set of inputs
		for(int i = 0; i < 16; i++) {
			while(Math.abs(CURRENT_ERROR) > ABSOLUTE_ERROR) {
			
			// Obtain outputs for hidden layer nodes
			for(int j = 0; j < 4; j++) {
				
				hiddenLayerOutputArray[j] = inputAndHiddenLayerWeights[0][j]*inputArray_1[i] 
						+ inputAndHiddenLayerWeights[1][j]*inputArray_2[i]
								+ inputAndHiddenLayerWeights[2][j]*inputArray_3[i]
										+ inputAndHiddenLayerWeights[3][j]*inputArray_4[i]
												+ hiddenLayerBias[j];
				
				sigmoidHiddenLayerArray[j] = (1/(1 + Math.exp(-hiddenLayerOutputArray[j])));
			}
			
			outputLayerBiasOutput = (1/1 + Math.exp(-outputLayerBias));
			
			// Obtain output from system of nodes		
			obtainedOutput = hiddenAndOutputLayerWeights[0][0]*sigmoidHiddenLayerArray[0]
					+ hiddenAndOutputLayerWeights[1][0]*sigmoidHiddenLayerArray[1]
							+ hiddenAndOutputLayerWeights[2][0]*sigmoidHiddenLayerArray[2]
									+ hiddenAndOutputLayerWeights[3][0]*sigmoidHiddenLayerArray[3]
											+ outputLayerBiasOutput;
			
			
			sigmoidOutputArray[i] = (1/(1 + Math.exp(-obtainedOutput)));
			
			CURRENT_ERROR = desiredOutputArray[i] - sigmoidOutputArray[i];
			
			// If the error is less than or equal to the threshold, then do not update
			if(!(Math.abs(CURRENT_ERROR) > ABSOLUTE_ERROR)) {
				System.out.println("Absolute Error: " + Math.abs(CURRENT_ERROR));
				CURRENT_ERROR = Double.MAX_VALUE;
				break;
			}
			epoch++;
			deltaK = sigmoidOutputArray[i]*(1 - sigmoidOutputArray[i])*(desiredOutputArray[i] - sigmoidOutputArray[i]);
			
				// Compute the weight difference for hidden-output layer
				for(int j = 0; j < 4; j++) {			
					hiddenAndOutputLayerWeightsDifference[j][0] = learningRate*deltaK*sigmoidOutputArray[j];
					if(momentumMode) {
						hiddenAndOutputLayerWeightsDifference[j][0] /= (1 - learningMomentumAmount);
					}
				}
				
				// Compute the weight difference for inner-hidden layer
				// Iterate over hidden layer
				for(int k = 0; k < 4; k++) {
					deltaJ = sigmoidHiddenLayerArray[k]*(1 - sigmoidHiddenLayerArray[k])*(hiddenAndOutputLayerWeights[k][0])*deltaK;
					// Iterate over inner layer
					for(int j = 0; j < 4; j++) {
						switch (j) {
						case 0:
							inputArray = inputArray_1.clone();
							break;
						case 1:
							inputArray = inputArray_2.clone();
							break;
						case 2:
							inputArray = inputArray_3.clone();
							break;
						case 3:
							inputArray = inputArray_4.clone();
							break;
						default:
							System.out.println("An unexpected error occurred!");
							System.exit(0);
							break;
							
						}

						inputAndHiddenLayerWeightsDifference[j][k] = learningRate*deltaJ*inputArray[i];
						if(momentumMode) {
							inputAndHiddenLayerWeightsDifference[j][k] /= (1 - learningMomentumAmount);
						}
					}
					hiddenLayerBiasDifference[k] = learningRate*deltaJ;
					if(momentumMode) {
						hiddenLayerBiasDifference[k] /= (1 - learningMomentumAmount);
					}
					
				}
				// Update all weights
				for(int k = 0; k < 4; k++) {
					hiddenAndOutputLayerWeights[k][0] = hiddenAndOutputLayerWeights[k][0] + hiddenAndOutputLayerWeightsDifference[k][0];
				}
				
				for(int k = 0; k < 4; k++) {
					for(int j = 0; j < 4; j++) {
						inputAndHiddenLayerWeights[j][k] = inputAndHiddenLayerWeights[j][k] + inputAndHiddenLayerWeightsDifference[j][k];
					}
				}
				
				// Update all biases
				outputLayerBiasDifference = learningRate*deltaK*outputLayerBias;
				if(momentumMode) {
					outputLayerBiasDifference /= (1 - learningMomentumAmount);
				}
				outputLayerBias = outputLayerBias + outputLayerBiasDifference;
				
				for(int k = 0; k < 4; k++) {
					hiddenLayerBias[k] = hiddenLayerBias[k] + hiddenLayerBiasDifference[k];
				}
			}
		}
	}
	
	public static void main(String args[]) {
		mlpnetwork mp = new mlpnetwork();
		mp.initializeValues();
		mp.startLearning();
		
		System.out.println("\nInput - Hidden layer weights");
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++) {
				System.out.println((i + 1) + "-" + (j + 1) + "   " + mp.inputAndHiddenLayerWeights[i][j]);
			}
		}
		
		System.out.println("\nHidden layer biases");
		for(int j = 0; j < 4; j++) {
			System.out.println((j + 1) + "   " + mp.hiddenLayerBias[j]);
		}
		
		System.out.println("\nHidden-output layer weights");
				for(int i = 0; i < 4; i++) {
					System.out.println((i + 1) + "   " + mp.hiddenAndOutputLayerWeights[i][0]);
				}
				
		System.out.println("\nOutput layer bias");
		System.out.println(mp.outputLayerBias);
						
		System.out.println("\nLearning Rate: " + mp.learningRate + " Epochs: " + mp.epoch);

	}
}