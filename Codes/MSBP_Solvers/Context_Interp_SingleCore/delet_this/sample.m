Avg_Data = readmatrix("ShortenedData.csv");
x = Avg_Data(:,1);
y = Avg_Data(:,2);

% Get coefficients of a line fit through the data.
coefficients = polyfit(x, y, 15);
% Create a new x axis with exactly 1000 points (or whatever you want).
xFit = linspace(min(x), max(x), 1000);
% Get the estimated yFit value for each of those 1000 new x locations.
yFit = polyval(coefficients , xFit);
% Plot everything.
plot(x, y, 'b.', 'MarkerSize', 15); % Plot training data.
hold on; % Set hold on so the next plot does not blow away the one we just drew.
plot(xFit, yFit, 'r-', 'LineWidth', 2); % Plot fitted line.
title("m=15");
grid on;

leastsquares_polyfit(x,y,1)
leastsquares_polyfit(x,y,2)
leastsquares_polyfit(x,y,3)

function c = leastsquares_polyfit(x,y,m)
    A = zeros(numel(y), m+1);
    for i=0:m
        A(:,i+1) = x.^i;
    end

    c = (A.'*A) \ (A.'*y);
end