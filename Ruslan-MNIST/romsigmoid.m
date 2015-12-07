function rom = romsigmoid(x)
y = logsig(x);
rom = y .* (1 - y);
end