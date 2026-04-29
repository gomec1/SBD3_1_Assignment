#####################################################
####     Week 4: Longitudinal Data (models)     #####
#####################################################


rm(list=ls())         #clear environment

#libraries
library(rio)          #to import all kinds of data
library(dplyr)        #for data manipulation
library(tidyr)        #dito
library(ggplot2)      #for making nice graphs
library(plm)          #for linear panel data models & Hausman (Durbin-Wu-Hausman) Test
library(naniar)

set.seed(1561)        #random seed for reproducibility

#change directory
getwd()
setwd("C:/Users/zgc3/OneDrive - Berner Fachhochschule/SBD3/2026/Week 4 - Practice & Longitudinal Data/Code&Data")
getwd()


#load the data
shp <- import("shp_wide.dta")

str(shp)
summary(shp)


###Recode some missing values
shp <- shp %>%
  mutate(across(where(is.numeric), ~ ifelse(. < 0, NA, .)))

#### Some visuals

##Plot personal income over time with average

#randomly select 10 observations
selected_ids <- sample(nrow(shp), 10)
df_selected <- shp[selected_ids, ] %>%
  mutate(id = factor(row_number()))  # create individual identifier



# Reshape to long format for selected individuals
df_long <- df_selected %>%
  select(id, iptotg_20, iptotg_21, iptotg_22, iptotg_23) %>%
  pivot_longer(
    cols = starts_with("iptotg_"),
    names_to = "year",
    values_to = "salary"
  ) %>%
  mutate(year = as.integer(paste0("20", gsub("iptotg_", "", year))))

# Compute overall average per year from full dataset
df_avg <- shp %>%
  summarise(
    across(c(iptotg_20, iptotg_21, iptotg_22, iptotg_23),
           \(x) mean(x, na.rm = TRUE))) %>%
  pivot_longer(
    cols = everything(),
    names_to = "year",
    values_to = "salary"
  ) %>%
  mutate(year = as.integer(paste0("20", gsub("iptotg_", "", year))))

# Plot
ggplot() +
  # Individual salary trajectories
  geom_line(
    data = df_long,
    aes(x = year, y = salary, group = id, color = id),
    alpha = 0.6, linewidth = 0.8
  ) +
  geom_point(
    data = df_long,
    aes(x = year, y = salary, group = id, color = id),
    size = 2, alpha = 0.7
  ) +
  # Overall average line
  geom_line(
    data = df_avg,
    aes(x = year, y = salary),
    color = "black", linewidth = 1.5, linetype = "dashed"
  ) +
  geom_point(
    data = df_avg,
    aes(x = year, y = salary),
    color = "black", size = 3, shape = 18
  ) +
  scale_x_continuous(breaks = c(2020, 2021, 2022, 2023)) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Individual Salary Trajectories vs. Overall Average",
    x = "Year",
    y = "Salary",
    color = "Individual",
    caption = "Dashed black line = overall sample average"
  ) +
  theme_minimal() +
  theme(legend.position = "right")



####Models



#For the models, we use the entire data set, but first need to reshape it to long format

#also, let's make our life easier and just take gender and education from 2020 as time-constant

shp <- shp %>% 
  mutate(gender = sex_20,
         education = case_when(
           educat_20 >= 0  & educat_20 <= 2  ~ "At most lower secondary",
           educat_20 >= 3  & educat_20 <= 6  ~ "Upper secondary",
           educat_20 >= 7  & educat_20 <= 10 ~ "Tertiary",
           .default = NA
         )) %>%
  select(-starts_with("sex_"),-starts_with("educat_"))

#drop minors
shp <- shp %>%
  filter(age_20>=18)


#reshape
shp_long <- shp %>%
  pivot_longer(
    cols = -c(idpers, gender, education),  # exclude time-constant variables
    names_to = c(".value", "year"),
    names_sep = "_",
    values_drop_na = FALSE
  ) %>%
  mutate(year = as.integer(paste0("20", year)))
  

#rename to more meaningful var names

shp_long <- shp_long %>%
  rename(life.satisfaction = pc44,
         height = pc45,
         weight = pc46,
         gross.pers.income = iptotg,
         partner = pd29,
         happy.partner = pf54)

#let's check duplicates

table(duplicated(shp_long))

#and NAs

gg_miss_var(shp_long)


### Let's use a very simple model to predict people's yearly gross income, based on education, gender, age, and weight

form <- gross.pers.income ~ education + gender + age + weight

#RE Model
re <- plm(form, data = shp_long, model = "random", effect = "individual")
summary(re)


#FE model
fe <- plm(form, data = shp_long, model = "within", effect = "individual")
summary(fe)  #What happend to the other predictors such as education and gender?

#Hausman test
phtest(fe, re)   #What does the significant test mean?


#Let's work a bit on that mincer equation

form.mincer <- gross.pers.income ~ education + gender + age + I(age^2) + weight
fe.mincer <- plm(form.mincer, data = shp_long, model = "within")
summary(fe.mincer)


##Hybrid model (mundlak correction)
hybrid_data <- shp_long %>%
  group_by(idpers) %>%
  mutate(
    mean_age = mean(age, na.rm = TRUE),
    mean_age.2 = mean(age*age, na.rm=TRUE),
    mean_weight = mean(weight, na.rm = TRUE)
  ) %>%
  ungroup()


hybrid <- plm(gross.pers.income ~ education + gender + age + I(age^2) + weight +
                mean_age + mean_age.2 + mean_weight, 
              data = hybrid_data, 
              model = "random", 
              effect = "individual")
summary(hybrid)   #Slightly different within estimates for age, age^2 and weight; why?

pdim(hybrid_data)     #we have an unbalanced panel! That's why.


#compare to simple OLS

summary(lm(gross.pers.income ~ education + gender + age + I(age^2) + weight,
           data = shp_long))
summary(hybrid)



###Wrap this in a ML context

##Problem: If we do a traditional split, the unit-level fixed or random effects are not defined for the testing data!

##Solution: If we have enough time points, we can do a temporal split (e.g. first three time-points for training, 
# making then predictions for the fourth one based on the training -- needs balanced panel for FE, RE not an issue 
# (shrinkage toward mean for unbalanced cases))


#Creating a balanced panel
shp_balanced <- shp_long %>%
  select(idpers, year, gross.pers.income, education, gender, age, weight) %>%
  drop_na() %>%
  group_by(idpers) %>%
  filter(n_distinct(year) == 4) %>%
  ungroup()

n_distinct(shp_balanced$idpers)  # number of units
table(shp_balanced$year)  

# 1. Temporal split 
train <- shp_balanced %>% filter(year != max(year))
test  <- shp_balanced %>% filter(year == max(year))

# Declare panel structure
train_p <- pdata.frame(train, index = c("idpers", "year"))
test_p  <- pdata.frame(test,  index = c("idpers", "year"))

#  2. Formula
form <- gross.pers.income ~ education + gender + age + weight

# 3. Fit models on training data 
re_model <- plm(form, data = train_p, model = "random")
fe_model <- plm(form, data = train_p, model = "within")

# 4. Predictions on testing data

test$pred_re <- predict(re_model, newdata = test)

# FE model: manually apply coefficients (no unit FE portable to test set)
fe_coefs    <- coef(fe_model)
test_matrix <- model.matrix(~ education + gender + age + weight, data = test)
test_matrix <- test_matrix[, -1, drop = FALSE]          # drop intercept
test_matrix <- test_matrix[, names(fe_coefs), drop = FALSE]
test$pred_fe <- as.numeric(test_matrix %*% fe_coefs)


#Evaluation
eval_metrics <- function(actual, predicted, model_name) {
  residuals <- actual - predicted
  data.frame(
    model = model_name,
    RMSE  = sqrt(mean(residuals^2, na.rm = TRUE)),
    MAE   = mean(abs(residuals),   na.rm = TRUE)
  )
}

results <- bind_rows(
  eval_metrics(test$gross.pers.income, test$pred_re, "Random Effects"),
  eval_metrics(test$gross.pers.income, test$pred_fe, "Fixed Effects")
)

print(results)


###########################################################
#########     Application to own research question     ####


##To do:

# 1. Plot a random sample (10) as well as the sample average of how people's life satisfaction evolves over 4 years
# 2. Estimate a RE and a FE model (based on your DAG, if possible)
# 3. Check whether endogeneity is an issue (Hausman test)
# 4. If it is, also estimate a hybrid model to keep the time-constant vars.
# 5. Optional: compare to a simple, pooled OLS model (using lm())