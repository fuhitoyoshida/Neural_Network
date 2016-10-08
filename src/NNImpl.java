/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes

	public ArrayList<Instance> trainingSet=null;//the training set

	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs

	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;

		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);

		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}

	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.5], it should return 2. 
	 * The parameter is a single instance. 
	 */

	//Round to 1 decimal only in calculateOutputForInstance(Instance inst)
	public int calculateOutputForInstance(Instance inst){
		//setup input values for input node with atttributes
		for(int i = 0; i < inst.attributes.size(); i++){
			inputNodes.get(i).setInput(inst.attributes.get(i));
		}
		//calculate ouputValue of hidden nodes
		for(Node hiddenNode : hiddenNodes){
			hiddenNode.calculateOutput();
		}
		//calculate outputValue of output nodes
		for(Node outputNode : outputNodes){
			outputNode.calculateOutput();
		}
		//find max outputValue of outputNodes
		double max = 0;
		int max_index = 0;
		for(int i = 0; i <  outputNodes.size(); i++){
			Node outputNode = outputNodes.get(i);
			//round to 1 decimal place
			double output = Math.round(outputNode.getOutput()*100)/100.0;
			if(output >= max){
				max = outputNode.getOutput();
				max_index = i;
			}
		}
		return max_index;
	}


	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */

	public void train(){
		double output = 0.0;
		List<Integer> teacher = null;
		double adjustedWeight = 0.0;
		double error = 0.0;
		double deltaK = 0.0;

		for(int counter = 0; counter < maxEpoch; counter++){
			for(Instance inst : trainingSet){
				calculateOutputForInstance(inst);
				teacher = inst.classValues;
				//jk weight
				for(int i = 0; i < outputNodes.size(); i++){
					Node kNode = outputNodes.get(i);
					output = kNode.getOutput();
					error = teacher.get(i) - output;
					deltaK = error*getReLU(kNode.getSum());
					for(int j = 0; j < kNode.parents.size(); j++){
						NodeWeightPair jkWeight = kNode.parents.get(j);
						Node jNode = jkWeight.node;
						adjustedWeight = getJK(jNode, deltaK);
						jkWeight.weight += adjustedWeight;

					}
				}
				//ij weight
				for(int i = 0; i < hiddenNodes.size(); i++){
					Node jNode = hiddenNodes.get(i);
					if(jNode.parents == null) continue;
					for(int j = 0; j < jNode.parents.size(); j++){
						NodeWeightPair ijWeight = jNode.parents.get(j);
						Node iNode = ijWeight.node;
						adjustedWeight = getIJ(iNode, jNode, teacher, i);
						ijWeight.weight += adjustedWeight;
					}
				}
			}
		}
	}

	private double getIJ(Node iNode, Node jNode, List<Integer> teacher, int index){
		double iOutput = iNode.getOutput();
		double jInput = jNode.getSum();
		double error = 0.0;
		double deltaK = 0.0;

		double sum = 0.0;
		for(int i = 0; i < outputNodes.size(); i++){
			Node kNode = outputNodes.get(i);
			error = teacher.get(i) - kNode.getOutput();
			deltaK = error*getReLU(kNode.getSum());
			NodeWeightPair jkWeight = kNode.parents.get(index);
			sum += jkWeight.weight*deltaK;
		}		
		return learningRate*iOutput*getReLU(jInput)*sum;
	}

	private double getJK(Node jNode, double deltaK){
		double jOutput = jNode.getOutput();
		return learningRate*jOutput*deltaK;
	}

	private double getReLU(double x){
		return x <= 0 ? 0 : 1;
	}

}


