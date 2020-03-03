clearvars
rng('shuffle');

learningrate = 0.07;
temperature = 30000;

w =-0.2 + (0.2+0.2)*rand(1,9);
disp(w);
if w(2) == 0
    w(2) = w(2)+0.01;
end
x = [0.1,0.1,-1;0.1,0.9,-1;0.9,0.1,-1;0.9,0.9,-1];
bias_value = [-1;-1;-1;-1];
%disp(input);
ye = [0.1,0.9,0.9,0.1];
y = zeros(4,1);
%disp(y)

Cost_Old = 0;
Cost_New = 0;
Probabilty = 0;

plot (x(1,1), x(1,2), 'k^')
hold on
plot (x(2,1), x(2,2), 'ks')
hold on
plot (x(3,1), x(3,2), 'ks')
hold on
plot (x(4,1), x(4,2), 'k^')
hold on


for n=1:2
for i=1:4
    if n == 1
    for j=1:3
        k(j) = x(i,j)*w(j);
    end
    Y_of_1(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
    else
        for j=1:3
            k(j) = x(i,j)*w(j+3);
        end
     Y_of_2(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
    end
end
end
YY = [Y_of_1.' Y_of_2.' bias_value];


for i=1:4
    for j=1:3
        k(j) = YY(i,j)*w(j+6);
    end
    y(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
end

Cost_Old = (ye(1)-y(1)^2) + (ye(2)-y(2)^2) + (ye(3)-y(3)^2) + (ye(4)-y(4)^2);


Random_Change = randi([1 9],1);

w(Random_Change) =-0.2 + (0.2+0.2)*rand(1); % For w2

for n=1:2
for i=1:4
    if n == 1
    for j=1:3
        k(j) = x(i,j)*w(j);
    end
    Y_of_1(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
    else
        for j=1:3
            k(j) = x(i,j)*w(j+3);
        end
     Y_of_2(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
    end
end
end
YY = [Y_of_1.' Y_of_2.' bias_value];

for i=1:4
    for j=1:3
        k(j) = YY(i,j)*w(j+6);
    end
    y(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
end

Cost_New = (ye(1)-y(1)^2) + (ye(2)-y(2)^2) + (ye(3)-y(3)^2) + (ye(4)-y(4)^2);


m=-3:0.1:3;
n=(-w(9)-w(7)*m)/w(8);
m=-3:0.1:3;
n=(-w(9)-w(7)*m)/w(8);
p2= plot(m,n,'k');
hold on

axis([-3 3 -3 3]);

for iteration = 1:temperature
    if Cost_New < 0.588888
        disp('breaking');
        break;
    end
    if Cost_Old >= Cost_New
        Cost_Old = Cost_New;
        Random_Change = randi([1 9],1);
 

w(Random_Change) = -0.2 + (0.2+0.2)*rand(1);
for n=1:2
for i=1:4
    if n == 1
    for j=1:3
        k(j) = x(i,j)*w(j);
    end
    Y_of_1(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
    else
        for j=1:3
            k(j) = x(i,j)*w(j+3);
        end
     Y_of_2(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
    end
end
end
YY = [Y_of_1.' Y_of_2.' bias_value];

for i=1:4
    for j=1:3
        k(j) = YY(i,j)*w(j+6);
    end
    y(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
end

Cost_New = (ye(1)-y(1)^2) + (ye(2)-y(2)^2) + (ye(3)-y(3)^2) + (ye(4)-y(4)^2);
    else
        Probability = exp((Cost_New-Cost_Old)/temperature);
        temp = 1-Probability;
        temperature = temperature - learningrate;
        if (Probability >= 0.50)
               Cost_Old = Cost_New;
               Random_Change = randi([7 9],1);
               
w(Random_Change) = -0.2 + (0.2+0.2)*rand(1);

for n=1:2
for i=1:4
    if n == 1
    for j=1:3
        k(j) = x(i,j)*w(j);
    end
    Y_of_1(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
   
    else
        for j=1:3
            k(j) = x(i,j)*w(j+3);
        end
     Y_of_2(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
    
    end
end
end
YY = [Y_of_1.' Y_of_2.' bias_value];

for i=1:4
    for j=1:3
        k(j) = YY(i,j)*w(j+6);
    end
    y(i) = sigmf(k(1)+k(2)+k(3),[1,0]);
end

Cost_New = (ye(1)-y(1)^2) + (ye(2)-y(2)^2) + (ye(3)-y(3)^2) + (ye(4)-y(4)^2);
        end
m=-3:0.1:3;
n=(-w(9)-w(7)*m)/w(8);

hold on
    end
end
disp(y)
m=-3:0.1:3;
n=(-w(9)-w(7)*m)/w(8);
p1= plot(m,n,'r');
hold on
axis([-3 3 -3 3]);

%m=-3:0.1:3;
%n=(-w(6)-w(4)*m)/w(5);
%p2= plot(m,n,'k');
%hold on

%axis([-3 3 -3 3]);