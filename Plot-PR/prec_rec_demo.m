x1 = rand(1000, 1);  
y1 = round(x1 + 0.5*(rand(1000,1) - 0.5));  
prec_rec(x1, y1);  
x2 = rand(1000,1);  
y2 = round(x2 + 0.75 * (rand(1000,1)-0.5));  
prec_rec(x2, y2, 'holdFigure', 1);  
legend('baseline','x1/y1','x2/y2','Location','SouthEast');  