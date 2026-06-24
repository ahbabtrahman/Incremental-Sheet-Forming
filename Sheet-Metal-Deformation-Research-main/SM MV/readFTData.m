%Read all csv files for each pass (Craft Files)
pass2 = readtable("Pass2.csv");
pass3 = readtable("Pass3.csv");
pass4 = readtable("Pass4.csv");
pass5 = readtable("Pass5.csv");
pass6 = readtable("Pass6.csv");
pass7 = readtable("Pass7.csv");
pass8 = readtable("Pass8.csv");

data = {pass2, pass3, pass4, pass5, pass6, pass7, pass8};

%Clear time bias
for i = 1:numel(data)
    timebias = data{i}(1,1);
    for k = 1:numel(data{i}(:,1))
        data{i}(k,1) = data{i}(k,1) - timebias;
    end
end
%% (3003 Files)
%Read all csv files for each pass
pass2 = readtable("Pass1.1.csv");
pass3 = readtable("Pass2.1.csv");
pass4 = readtable("Pass3.1.csv");
pass5 = readtable("Pass4.1.csv");
pass6 = readtable("Pass5.1.csv");
pass7 = readtable("Pass6.1.csv");
pass8 = readtable("Pass7.1.csv");

data = {pass2, pass3, pass4, pass5, pass6, pass7, pass8};

%Clear time bias
for i = 1:numel(data)
    timebias = data{i}(1,1);
    for k = 1:numel(data{i}(:,1))
        data{i}(k,1) = data{i}(k,1) - timebias;
    end
end

%% Plot forces

evencount = 0;
oddcount = 0;
for i = 1:numel(data)
    if mod(i+1,2) == 0
        evencount = evencount + 1;
        figure(1);
        subplot(4,1,evencount)
        plot(data{i}, "Var1", "Var2");
        depth = -.04*(i+1);
        xlabel("Time (s)")
        ylabel("Force (N)")
        title("Force Feedback on Depth = " + string(depth) +"mm")
        refline(0,0);
    else
        oddcount = oddcount + 1;
        figure(2);
        subplot(3,1,oddcount)
        plot(data{i}, "Var1", "Var2");
        depth = -.04*(i+1);
        xlabel("Time (s)")
        ylabel("Force (N)")
        title("Force Feedback on Depth = " + string(depth) +"mm")
        refline(0,0)
    end
end
figure(1);
sgtitle("Even Passes", 'Fontsize', 10)
figure(2);
sgtitle("Odd Passes",'Fontsize', 10)

%% Relationship between Depth and Force
zheight = [ -.08, -.12,-.16, -.2, -.24, -.28, -.32];
%Referencing from graph
pass = [2,3,4,5,6,7,8];
ref1 = [-1.355, -1.046,-1.045,-1.646,-1.918,-1.689,-2.588];
figure;
plot(pass,ref1)
