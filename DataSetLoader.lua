
require 'csvigo'
local num2char = require 'num2char'
local char2num = require 'char2num'

local DataSetLoader = {}
DataSetLoader.__index = DataSetLoader


function DataSetLoader.create(filename, batchSize)
	local self = {}
	setmetatable(self, DataSetLoader)
	self.filename = filename
	self.batchSize = batchSize
	self:load(filename)
	self:createBatches(batchSize)
	self.currentBatch = 0
	self.evaluatedBatches = 0
	return self
end

function DataSetLoader:load(filename)
	self.data = csvigo.load({path=filename, verbose=false})
	assert(#self.data.questions == #self.data.answers)
	self.data.size = #self.data.questions
	return self.data
end

function DataSetLoader:createBatches(batchSize)
	local X, Y = {}, {}
	local k = 1
	for i=1, self.data.size, batchSize do
		if self.data.size - i < batchSize then break end
		local qmax, amax = 0, 0
		local q, a = {}, {}
		for j=1, batchSize do
			q[j] = self.string2tensor(self.data.questions[i])
			a[j] = self.string2tensor(self.data.answers[i])
			qmax = math.max(q[j]:size(1), qmax)
			amax = math.max(a[j]:size(1), amax)
		end
		X[k] = torch.ByteTensor(batchSize, qmax)
		Y[k] = torch.ByteTensor(batchSize, amax)
		for j=1, batchSize do
			X[k][j]:sub(1, q[j]:size(1)):copy(q[j])
			Y[k][j]:sub(1, a[j]:size(1)):copy(a[j])
		end
	end
	self.X = X
	self.Y = Y
end

function DataSetLoader:nextBatch()
	self.currentBatch = (self.currentBatch % #self.X) + 1
	self.evaluatedBatches = self.evaluatedBatches + 1
	return self.X[self.currentBatch], self.Y[self.currentBatch]
end

function DataSetLoader.string2tensor(s)
	local t = torch.ByteTensor(#s+1)
	for i=1, #s do
		local num = char2num[s:sub(i, i)]
		t[i] = tonumber(num)
	end
	t[-1] = -1 --stop signal
	return t
end

function DataSetLoader.tensor2string(t)
	local s = {}
	for i=1, t:size(1)-1 do
		s[i] = num2char[t[i]]
	end
	return table.concat(s)
end

return DataSetLoader