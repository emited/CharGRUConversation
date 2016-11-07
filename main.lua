
require 'csvigo'
local num2char = require 'num2char'
local char2num = require 'char2num'



function char2tensor(s)
	local t = torch.ByteTensor(#s)
	for i=1, #s do
		local num = char2num[s:sub(i, i)]
		t[i] = tonumber(num)
	end
	return t
end

function tensor2char(t)
	local s = {}
	for i=1, t:size(1) do
		s[i] = num2char[t[i]]
	end
	return table.concat(s)
end


local csv = csvigo.load('data/qa.csv')
s = csv.questions[1]
t = char2tensor(s)
print(tensor2char(t))