
require 'optim'
local model_utils = require 'model_utils'
local num2char = require 'num2char'

local opt = {
	filename = 'data/qa.csv',
	model = '',
	batchSize = 1000,
	seed = 123,
	saveEvery = 1000,
	printEvery = 1,
}

torch.manualSeed(opt.seed)
protos = torch.load(opt.model)