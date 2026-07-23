library(dplyr)
df = read.csv("/Users/natanon/Documents/CICM/Hokkaido University/Dataset Analyze/datasets/asd/out.csv")
head(df)
df$processed_measurement
df$affinity_type
unique(df$affinity_type)
table(df$affinity_type)

df_bool=filter(df,affinity_type=="bool")
df_bool
df$affinity_type
table(df_bool$affinity)
df

g_bool=df_bool %>% group_by(antigen_sequence) %>% summarize(num=n(), num_heavy=length(heavy_sequence), num_light=length(light_sequence))
g_bool
head(g_bool)
g_bool$num
g_bool$num_heavy
df_bool$light_sequence

nrow(df_bool)

df_bool_bind = filter(df_bool,affinity=="1.0")
df_bool_nobind = filter(df_bool,affinity=="0.0")
df_bool_bind_count = df_bool_bind %>% group_by(antigen_sequence) %>% summarize(num=n()) %>% arrange(desc(num))
df_bool_nobind_count = df_bool_nobind %>% group_by(antigen_sequence) %>% summarize(num=n()) %>% arrange(desc(num))
df_bool_bind_count
df_bool_nobind_cout

hist(df_bool_bind_count$num)
h=hist(df_bool_bind_count$num, breaks=1000)
names(h)
plot(h$mids,h$counts,log="xy", ty="b")

h2=hist(df_bool_nobind_count$num, breaks=1000)
names(h2)
plot(h$mids,h$counts,log="xy", ty="b")

lines(h2$mids,h2$counts,log="xy",ty="b",pch=2)

df_fuzzy=filter(df,affinity_type=="fuzzy")
df_fuzzy$affinity


unique(df_fuzzy$affinity)

table(df$affinity_type)



