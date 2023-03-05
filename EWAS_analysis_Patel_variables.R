library(hash)

# some variables are in multiple tabs like URXOP1

response_vars <- hash()
response_vars[["custom_variables_by_CCNR"]] <- c(
  't2d', 'metabolic_syndrome', 'metabolic_syndrome_examination', 
  # 'framingham_risk_10_years', 
  'ascvd_10y_frs_simple_lab', 'ascvd_10y_frs_lab',
  'ascvd_10y_accaha_lab', 'LBXACR_lab_detectable',
  'LBXGLY_lab_detectable'
)

response_vars[["Acrylamide_and_Glycidamide"]] <- c('LBXACR', 'LBXGLY')

response_vars[["Alcohol_Consumption"]] <- c('drink_five_per_day', 'total_days_drink_year', 'total_days_5drink_year', 'quantity_drink_per_day')

response_vars[["Allergen_IgE_Serum_Tests"]] <- c('LBXE72', 'LBXE74', 'LBXIW1', 'LBXW11', 'LBXIG2', 'LBXIM3', 'LBXF13', 'LBXID1', 'LBXIG5', 'LBXIE5', 'LBXIM6', 'LBXIF2', 'LBXID2', 'LBXII6', 'LBXIF1', 'LBXIGE', 'LBXIE1', 'LBXF24', 'LBXIT7', 'LBXIT3')

response_vars[["Any_Disease"]] <- c('testis_cancer_self_report', 'current_asthma', 'bladder_cancer_self_report', 'gallbladder_cancer_self_report', 'ever_osteo_arthritis', 'ever_rheumatoid_arthritis', 'thyroid_cancer_self_report', 'any_ht', 'any_family_cad', 'blood_cancer_self_report', 'uterine_cancer_self_report', 'pancreatic_cancer_self_report', 'ovarian_cancer_self_report', 'colon_cancer_self_report', 'prostate_cancer_self_report', 'melanoma_self_report', 'esophagus_cancer_self_report', 'leukemia_self_report', 'lymphoma_self_report', 'skin_cancer_self_report', 'rectum_cancer_self_report', 'lung_cancer_self_report', 'cervix_cacner_self_report', 'soft_cancer_self_report', 'ever_arthritis', 'cad', 'breast_cancer_self_report', 'kidney_cancer_self_report', 'mouth_cancer_self_report', 'other_cancer_self_report', 'other_skin_cancer_self_report', 'nervous_cancer_self_report', 'ever_asthma', 'bone_cancer_self_report', 'larynx_cancer_self_report', 'liver_cancer_self_report', 'stomach_cancer_self_report', 'any_cancer_self_report', 'brain_cancer_self_report', 'any_diabetes')

response_vars[["Biochemistry_Profile"]] <- c('LBXSTB', 'LBXSGL', 'LBXSAL', 'LBXSIR', 'LBXSAPSI', 'LBXSNASI', 'LBXSTP', 'LBXSASSI', 'LBXSATSI', 'LBXSGB', 'LBXSKSI', 'LBXSC3SI', 'LBXSGTSI', 'LBXSTR', 'LBXSCRINV', 'LBXSCLSI', 'LBXSUA', 'LBXSCH', 'LBXSCA', 'LBXSCR', 'LBXSPH', 'LBXSOSSI', 'LBXSLDSI', 'LBXSBU')

response_vars[["Biochemistry_Profile_and_Hormones"]] <- c('LBXSTB', 'LBXSGL', 'LBXSIR', 'LBXSAL', 'LBXSAPSI', 'LBXLH', 'LBXSNASI', 'LBXSTP', 'LBXSASSI', 'LBXSATSI', 'LBXSGB', 'LBXSKSI', 'LBXSC3SI', 'LBXSGTSI', 'LBXSTR', 'LBXSCRINV', 'LBXSCLSI', 'LBXSUA', 'LBXSCH', 'LBXSCA', 'LBXSCR', 'LBXSPH', 'LBXSOSSI', 'LBXSLDSI', 'LBXFSH', 'LBXSBU')

response_vars[["Biochemistry_Profile_Follicle_Stimulating_Hormone_and_Luteinizing_Hormone"]] <- c('LBXSTB', 'LBXSGL', 'LBXSAL', 'LBXSIR', 'LBXSAPSI', 'LBXLH', 'LBXSNASI', 'LBXSTP', 'LBXSASSI', 'LBXSATSI', 'LBXTSH', 'LBXSGB', 'LBXSKSI', 'LBXSC3SI', 'LBXSGTSI', 'LBXSTR', 'LBXT4', 'LBXSCRINV', 'LBXSCLSI', 'LBXSUA', 'LBXSCH', 'LBXSCA', 'LBXSCR', 'LBXSPH', 'LBXSOSSI', 'LBXSLDSI', 'LBXFSH', 'LBXSBU')

response_vars[["Blood_Lead_and_Blood_Cadmium"]] <- c('LBXBPB', 'LBXBCD')

response_vars[["Blood_Lead_Cadmium_Mercury"]] <- c('LBXBPB', 'LBXBCD', 'LBXIHG', 'LBXTHG')

response_vars[["Blood_Pressure"]] <- c('BPXPLS', 'MSYS', 'BPXCHR', 'MDIS')

response_vars[["Body_Measurements"]] <- c('BMXWAIST', 'BMXTHICR', 'BMXTRI', 'BMXWT', 'BMXSUB', 'BMXRECUM', 'BMXHT', 'BMXCALF', 'BMXLEG', 'BMXHEAD')

response_vars[["Body_Measures"]] <- c('BMXSUB', 'BMXHT', 'DXDTOLE', 'DXDTOFAT', 'BMXTHICR', 'BMXWAIST', 'DXXHEBMD', 'BMXBMI', 'BMXTRI', 'DXDTOBMD', 'DXXPEBMD', 'DXXLSBMD', 'DXDTRLE', 'DXXTRFAT', 'BMXLEG', 'BMXWT', 'BMXRECUM', 'BMXCALF', 'BMXHEAD')

response_vars[["C_Reactive_Protein"]] <- c(
  'LBXCRP', 'LBXFB', 'URXNT', 'LBXPT21', 'LBXBAP', 'LBXHP1'
  )

#response_vars[["C_reactive_protein"]] <- c('LBXHP1', 'LBXCRP', 'URXNT', 'LBXFB', 'LBXBAP')

response_vars[["Cardiovascular_Fitness"]] <- c('CVDS1DI', 'CVDR1SY', 'CVDVOMAX', 'CVDS1SY', 'CVDS2DI', 'CVDESVO2', 'CVDR2SY', 'CVDR1HR', 'CVDR2HR', 'CVDR2DI', 'CVDR1DI', 'CVDS1HR', 'CVDS2HR', 'CVDS2SY')

response_vars[["Chlamydia_and_Gonorrhea"]] <- c('URXUGC', 'URXUCL')

response_vars[["Cognitive_Functioning"]] <- c('CFDFINSH', 'CFDRIGHT')

response_vars[["Complete_Blood_Count"]] <- c('LBXPLTSI', 'LBDNENO', 'LBDLYMNO', 'LBXEOPCT', 'LBXHGB', 'LBDMONO', 'LBXRDW', 'LBDBANO', 'LBXHCT', 'LBDEONO', 'LBXMPSI', 'LBXMCHSI', 'LBXMCVSI', 'LBXRBCSI', 'LBXMC', 'LBXLYPCT', 'LBXNEPCT', 'LBXWBCSI', 'LBXBAPCT', 'LBXMOPCT')

response_vars[["Cotinine"]] <- c('LBXCOT')

response_vars[["Cryptosporidium_and_Toxoplasma"]] <- c('LBXTO2', 'LBXTO5', 'LBXTO3', 'LBXTO1')

response_vars[["Cryptosporidum_and_Toxoplasma"]] <- c('LBDC1', 'LBXTO2', 'LBXTO3', 'LBDC2', 'LBXTO1')

response_vars[["Dioxins"]] <- c('LBXF06', 'LBX138', 'LBX099', 'LBX146', 'LBXF03', 'LBXHPE', 'LBX128', 'LBXD01', 'LBXOXY', 'LBX170', 'LBXPCB', 'LBXTNA', 'LBXF04', 'LBX074', 'LBX183', 'LBXTCD', 'LBXD07', 'LBXF10', 'LBXPDE', 'LBX066', 'LBX178', 'LBXF02', 'LBX187', 'LBXODT', 'LBXPDT', 'LBXHXC', 'LBXD05', 'LBX153', 'LBXF08', 'LBX172', 'LBXGHC', 'LBX180', 'LBXD03', 'LBXHCB', 'LBX052', 'LBX028', 'LBXTC2', 'LBX101', 'LBXBHC', 'LBX177', 'LBXD04', 'LBXF07', 'LBX167', 'LBX156', 'LBXF01', 'LBXF05', 'LBX105', 'LBX157', 'LBXMIR', 'LBX118')

response_vars[["Dioxins_and_Other_Persistent_Organochlorines_Furans_and_Co__Planar_PCBs"]] <- c('LBXF06', 'LBX138', 'LBX099', 'LBX146', 'LBXF03', 'LBXHPE', 'LBX151', 'LBX128', 'LBXD01', 'LBXOXY', 'LBX170', 'LBXPCB', 'LBXTNA', 'LBXF04', 'LBX074', 'LBX183', 'LBXTCD', 'LBXD07', 'LBXF10', 'LBXALD', 'LBXPDE', 'LBX066', 'LBX178', 'LBX189', 'LBXF02', 'LBXEND', 'LBX187', 'LBX149', 'LBXODT', 'LBX194', 'LBXPDT', 'LBX087', 'LBXHXC', 'LBXDIE', 'LBXD05', 'LBX153', 'LBXF08', 'LBX206', 'LBX172', 'LBXD02', 'LBXGHC', 'LBX180', 'LBXD03', 'LBXHCB', 'LBX052', 'LBXTC2', 'LBX101', 'LBXBHC', 'LBX177', 'LBX195', 'LBXD04', 'LBXF07', 'LBX110', 'LBX167', 'LBD199', 'LBX156', 'LBXF01', 'LBXF05', 'LBX105', 'LBX157', 'LBX196', 'LBXMIR', 'LBXF09', 'LBX118')

response_vars[["Dioxins_Furans_and_Coplanar_PCBs"]] <- c('LBXF06', 'LBXF03', 'LBXD01', 'LBXPCB', 'LBXF04', 'LBX074', 'LBXTCD', 'LBXD07', 'LBXF10', 'LBX066', 'LBX189', 'LBXF02', 'LBXHXC', 'LBXD05', 'LBXF08', 'LBXD02', 'LBXD03', 'LBX028', 'LBXTC2', 'LBXD04', 'LBXF07', 'LBX167', 'LBX156', 'LBXF01', 'LBXF05', 'LBX105', 'LBX157', 'LBXF09', 'LBX118')

response_vars[["Environmental_Pesticides"]] <- c('URX14D', 'URXOPP', 'URX1TB', 'URXDCB', 'URX3TB')

#response_vars[["Environmental_Phenols"]] <- c('URX4TO', 'URXTRS', 'URXBPH', 'URXBP3')

response_vars[["Environmental_phenols"]] <- c('URXEPB', 'URXBP3', 'URX4TO', 'URXPPB', 'URXBUP', 'URXBPH', 'URXMPB', 'URXTRS')

response_vars[["Erythrocyte_RBC_Folate_and_Serum_Folate"]] <- c('LBXRBF', 'LBXFOL')

#response_vars[["Erythrocyte_Protoporphyrin"]] <- c('LBXEPP')

#response_vars[["Erythrocyte_Protoporphyrin_EPP"]] <- c('LBXEPP')

response_vars[["Erythrocyte_Protoporphyrin_and_Selenium"]] <- c('LBXSEL', 'LBXEPP')

response_vars[["Family_Smoking"]] <- c('SMD415B', 'SMD450', 'SMD440', 'SMD410', 'SMD415A', 'SMD415', 'SMD430', 'SMD415C')

#response_vars[["Ferritin"]] <- c('LBXFER')

response_vars[["Ferritin_and_Transferrin_Receptor"]] <- c('LBXTFR', 'LBXFER')

response_vars[["Food_Consumption"]] <- c('DR1TPROT', 'DR1TM161', 'DR1TP182', 'DRD350GQ', 'DRD370D', 'DR1_330', 'DR1TATOA', 'DRD370CQ', 'DR1TP183', 'DR1_330Z', 'DR1TCALC', 'DR1TACAR', 'DRDRESP', 'DRDSDT7', 'DRD370PQ', 'DR1TCAFF', 'DRD370O', 'DRD370EQ', 'DR1TP204', 'salt_substitute', 'DRD370G', 'DRD370N', 'DRD350F', 'DR1TVARE', 'DRDSDT5', 'DRD370QQ', 'DR1TMOIS', 'DRD370P', 'DR1TSODI', 'DR1TP205', 'DRD370I', 'DR1TCARB', 'DRD350HQ', 'DRD370A', 'DRD370DQ', 'DRD370UQ', 'DR1TB12A', 'DR1TP184', 'ordinary_salt', 'DR1TCHOL', 'DRD370RQ', 'DRD370Q', 'DR1TP225', 'DR1TPFAT', 'DRD370BQ', 'DRD350A', 'DRD370HQ', 'DRD350IQ', 'DRD370OQ', 'spring_water', 'DR1TVARA', 'DRDSDT8', 'DR1TCHL', 'DRDTSODI', 'well_water', 'DR1TTHEO', 'DR1TVK', 'DR1TM181', 'DRD370C', 'DR1TMFAT', 'DRD370JQ', 'DR1TS040', 'DRD350E', 'DR1TLYCO', 'DRD350G', 'DR1TVB12', 'no_salt', 'DR1TRET', 'DRD370K', 'DRD370B', 'DRD370H', 'DRD370LQ', 'DR1TS180', 'DRD350J', 'DR1_320', 'DR1TM201', 'DR1TCRYP', 'DR1TS160', 'DRD340', 'DRD350CQ', 'DRD370IQ', 'DR1TLZ', 'DRD330GW', 'DRD350C', 'DR1TIRON', 'DRD370F', 'DRD350JQ', 'DRD370S', 'DR1TZINC', 'DR1TS100', 'DRD350B', 'DRDSDT1', 'DR1TPOTA', 'DRD370NQ', 'lite_salt', 'DRDSDT6', 'DR1TSELE', 'DR1BWATR', 'DRDSDT3', 'DR1TATOC', 'DRD370KQ', 'DR1TFA', 'DRD370R', 'DRD350H', 'DRD370M', 'DR1TS060', 'DRDCWATR', 'DR1TS140', 'DRD350AQ', 'DR1TVB1', 'DRD350D', 'DRD370TQ', 'DR1TPHOS', 'DR1TS120', 'DRD370U', 'DR1TTFAT', 'DRD370T', 'DR1TALCO', 'DR1TFDFE', 'DRDSDT2', 'DR1TCARO', 'DRD370MQ', 'DR1TVE', 'DRDSDT4', 'DR1CWATR', 'DRD350EQ', 'DRD320GW', 'DBQ095', 'DRD370AQ', 'DRD350BQ', 'DR1TMAGN', 'DR1TVB6', 'DR1TFF', 'DRD370J', 'DRD360', 'community_supply', 'DRD350K', 'DR1TWATE', 'DR1TVB2', 'DR1TCOPP', 'DR1TKCAL', 'DR1TP226', 'DRD370SQ', 'DR1TFIBE', 'DR1TSFAT', 'DRD350FQ', 'DRD370L', 'DRD370GQ', 'DR1TSUGR', 'DR1TS080', 'DRD370E', 'DR1TVC', 'DR1TNIAC', 'DRDDRSTZ', 'DBD100', 'DR1TBCAR', 'DRD350DQ', 'DRD350I', 'DRD370V', 'DRD370FQ', 'DR1TVAIU', 'DR1TM221')

response_vars[["Glycohemoglobin"]] <- c('LBXGH')

response_vars[["HDL_Cholesterol"]] <- c('LBDHDD')

response_vars[["HIV"]] <- c('LBDHI', 'LBXCD4', 'LBXCD8')

response_vars[["HPV_Serum"]] <- c('LBXS06MK', 'LBXS18MK', 'LBXS11MK', 'LBXS16MK')

response_vars[["Hair_Mercury"]] <- c('HRDHG', 'HRXHG')

response_vars[["Heavy_Metals"]] <- c('URXUBA', 'URXUTU', 'URXUCS', 'URXUBE', 'URXUPT', 'URXUCD', 'URXUMO', 'URXUPB', 'URXUTL', 'URXUCO', 'URXUSB', 'URXUUR')

response_vars[["Hepatitis"]] <- c('LBDHBG', 'LBXHBC', 'LBDHD', 'LBDHCV')

response_vars[["Hepatitis_A"]] <- c('LBXHA')

#response_vars[["Hepatitis_A_Antibody"]] <- c('LBXHA')

#response_vars[["Hepatitis_A_Antibody_"]] <- c('LBXHA')

#response_vars[["Hepatitis_A_B_C_and_D"]] <- c('LBDHBG', 'LBXHBC', 'LBDHD', 'LBDHCV')

response_vars[["Hepatitis_B_Surface_Antibody"]] <- c('LBXHBS')

#response_vars[["Hepatitis_B_surface_antibody"]] <- c('LBXHBS')

#response_vars[["Hepatitis_B_and_D"]] <- c('LBDHBG', 'LBXHBC', 'LBDHD')

#response_vars[["Hepatitis_C_antibody"]] <- c('LBDHCV')

response_vars[["Herpes_I_and_II"]] <- c('LBXHE2', 'LBXHE1')

response_vars[["Herpes_Simplex_Virus_I_and_II"]] <- c('LBXHE2', 'LBXHE1')

response_vars[["Homocysteine"]] <- c('LBXHCY')

response_vars[["Homocysteine_and_MMA"]] <- c('LBXMMA', 'LBXHCY')

response_vars[["Housing_Characteristics"]] <- c('home_painted_12mos', 'paint_chipping_outside', 'old_paint_scraped', 'use_water_treatment', 'how_many_years_in_house', 'paint_chipping_inside', 'private_water_source', 'house_type', 'house_age')

response_vars[["Human_Immunodeficiency_Virus"]] <- c('LBXCD4', 'LBDHI', 'LBXCD8')

response_vars[["Human_Immunodeficiency_Virus_HIV"]] <- c('LBDHI')

response_vars[["Immunization"]] <- c('hepa', 'pneu', 'hepb')

response_vars[["Iron_and_TIBC"]] <- c('LBDPCT', 'LBXIRN', 'LBXTIB')

response_vars[["Iron_TIBC_Transferrin"]] <- c('LBDPCT', 'LBXIRN', 'LBXTIB')

response_vars[["Iron_TIBC_Transferrin_Saturation"]] <- c('LBDPCT', 'LBXIRN', 'LBXTIB')

response_vars[["Lead_Dust"]] <- c('LBDDWS', 'LBXDFSF', 'LBXDFS')

response_vars[["Measles_Rubella_and_Varicella"]] <- c('LBXVAR', 'LBDRUIU', 'LBXME')

response_vars[["Melamine_Surplus_Urine"]] <- c('SSMEL')

response_vars[["Methicillin_Resistant_Staphylococcus_Aureus"]] <- c('LBXM1', 'LBXETA', 'LBXMI1', 'LBXMF1', 'LBXMP1', 'LBXDY1', 'LBXCD1', 'LBXMD1', 'LBXETH', 'LBXMZ1', 'LBXMO1', 'LBXETC', 'LBXMR1', 'LBXETD', 'LBXMG1', 'LBXML1', 'LBXMY1', 'LBXCH1', 'LBXPVL', 'LBXTSS', 'LBXMS1', 'LBXETB', 'LBXMT1', 'LBXME1', 'LBXMC1')

response_vars[["Non_dioxin_like_Polychlorinated_Biphenyls"]] <- c('LBX138', 'LBX099', 'LBX146', 'LBX151', 'LBX128', 'LBX170', 'LBX183', 'LBX209', 'LBX044', 'LBX178', 'LBX187', 'LBX149', 'LBX194', 'LBX087', 'LBX153', 'LBX206', 'LBX172', 'LBX180', 'LBX052', 'LBX101', 'LBX177', 'LBX195', 'LBX049', 'LBX110', 'LBD199', 'LBX196')

response_vars[["Nutritional_Biochemistries"]] <- c('LBXIHG', 'LBXSEL', 'LBXFOL', 'LBXIRN', 'LBXMMA', 'LBXBCD', 'LBXB12', 'LBXBPB', 'LBXRPL', 'URXUHG', 'LBXGTC', 'LBXTHG', 'LBXRST', 'LBXEPP', 'LBXVIA', 'LBDPCT', 'LBXTIB', 'LBXRBF', 'LBXFER', 'LBXVIE', 'LBXHCY', 'LBXCOT')

response_vars[["Occupation"]] <- c('occupation_farm', 'current_loud_noise', 'num_months_main_job', 'occupation_laborer', 'occupation_military', 'industry_construction', 'occupation_transport', 'industry_mining', 'num_months_longest_job', 'occupation_construction', 'ever_loud_noise_gt3', 'ever_loud_noise_gt3_2', 'smell_tobacco', 'occupation_repair', 'industry_agriculture', 'industry_transport', 'occupation_household', 'occupation_midmanage', 'industry_other', 'industry_manufacturing', 'occupation')

response_vars[["Organochlorine_Pesticides"]] <- c('LBXGHC', 'LBXHPE', 'LBXALD', 'LBXPDE', 'LBXEND', 'LBXHCB', 'LBXODT', 'LBXPDT', 'LBXDIE', 'LBXOXY', 'LBXMIR', 'LBXTNA', 'LBXBHC')

response_vars[["PHPYPA_Urinary_Phthalates"]] <- c('URXP07', 'URXMIB', 'URXP19', 'URXP06', 'URXP15', 'URXMEP', 'URXP17', 'URXP18', 'URXP09', 'URXP02', 'URXMCP', 'URXMHH', 'URXMNM', 'URXMZP', 'URXMBP', 'URXP24', 'URXP10', 'URXP13', 'URXMNP', 'URXP21', 'URXP04', 'URXMOP', 'URXP08', 'URXP16', 'URXP01', 'URXDMA', 'URXP22', 'URXMC1', 'URXP20', 'URXEQU', 'URXETD', 'URXP05', 'URXMOH', 'URXP14', 'URXP12', 'URXDAZ', 'URXGNS', 'URXP03', 'URXMHP', 'URXETL', 'URXP11')

response_vars[["PSA_and_Questions"]] <- c('LBXP1', 'LBXP2', 'LBDP3')

response_vars[["Parathyroid_Hormone"]] <- c('LBXPT21')

response_vars[["Perchlorate_Nitrate_and_Iodide_in_Tap_Water"]] <- c('LBXWP8', 'LBXWNO', 'LBXWIO')

response_vars[["Perchlorate_Urinary_Nitrate_Urinary_Thiocyanate"]] <- c('URXSCN', 'URXNO3', 'URXUP8')

response_vars[["Perfluorinated_Chemicals_Surplus_Sera_SSPFC_A"]] <- c('LBXMPAH', 'LBXPFHS', 'LBXPFDO', 'LBXPFOA', 'LBXPFSA', 'LBXPFNA', 'LBXPFOS', 'LBXPFUA', 'LBXPFDE', 'LBXEPAH', 'LBXPFHP')

response_vars[["Pesticides"]] <- c('URXOP5', 'URX24D', 'URXCPM', 'URXOPM', 'URXOP3', 'URX3TB', 'URXCCC', 'URXCB3', 'URXCMH', 'URXDIZ', 'URXPPX', 'URXDPY', 'URXOP2', 'URXACE', 'URX14D', 'URXOP1', 'URX1TB', 'URXTCC', 'URXDEE', 'URX4FP', 'URXCBF', 'URXATZ', 'URXPCP', 'URX25T', 'URXOP6', 'URXMET', 'URXOPP', 'URXMAL', 'URXPAR', 'URXOP4')

response_vars[["Pharmaceutical_Use"]] <- c('MEDROXYPROGESTERONE_ACETATE', 'PIOGLITAZONE', 'HYDROCHLOROTHIAZIDE__TRIAMTERENE', 'ENALAPRIL_MALEATE', 'HYDROXYZINE', 'QUINAPRIL', 'DILTIAZEM', 'ESOMEPRAZOLE', 'METOPROLOL_SUCCINATE', 'IRBESARTAN', 'HYDROCHLOROTHIAZIDE__LISINOPRIL', 'WARFARIN_SODIUM', 'GLYBURIDE__METFORMIN', 'DOXAZOSIN_MESYLATE', 'HYDROCHLOROTHIAZIDE', 'LOSARTAN_POTASSIUM', 'AZITHROMYCIN_DIHYDRATE', 'DIAZEPAM', 'DOXAZOSIN', 'ESTRADIOL', 'GLIMEPIRIDE', 'THEOPHYLLINE', 'QUINAPRIL_HYDROCHLORIDE', 'FLUOXETINE_HYDROCHLORIDE', 'TRAZODONE', 'METOPROLOL_TARTRATE', 'ISOSORBIDE_MONONITRATE', 'BUDESONIDE', 'IPRATROPIUM_BROMIDE', 'ETHINYL_ESTRADIOL__NORGESTIMATE', 'AMOXICILLIN_TRIHYDRATE__CLAVULANATE_POTASSIUM', 'LATANOPROST_OPHTHALMIC', 'AMOXICILLIN__CLAVULANATE', 'ATENOLOL', 'CROMOLYN_SODIUM', 'ANTIBIOTIC_UNSPECIFIED', 'POLYETHYLENE_GLYCOL_3350', 'GLYBURIDE', 'HYDROCHLOROTHIAZIDE__VALSARTAN', 'POTASSIUM_CHLORIDE', 'ACETAMINOPHEN__PROPOXYPHENE_NAPSYLATE', 'AMOXICILLIN', 'FENOFIBRATE', 'BUPROPION', 'CLARITHROMYCIN', 'LORATADINE', 'PANTOPRAZOLE_SODIUM', 'HYDROCODONE_UNSPECIFIED', 'ALLOPURINOL', 'VENLAFAXINE_HYDROCHLORIDE', 'GLIPIZIDE', 'ROFECOXIB', 'FLUTICASONE_PROPIONATE', 'FLUTICASONE', 'METOPROLOL', 'METFORMIN', '99999', 'RANITIDINE', 'ACETAMINOPHEN__CODEINE', 'PAROXETINE_HYDROCHLORIDE', 'PREDNISONE', 'TRAMADOL', 'CIMETIDINE_HYDROCHLORIDE', 'VALSARTAN', 'MONTELUKAST', 'GABAPENTIN', 'OXYBUTYNIN', 'CLONAZEPAM', 'ALBUTEROL', 'CONJUGATED_ESTROGENS', 'LORAZEPAM', 'AMLODIPINE__BENAZEPRIL', 'BENAZEPRIL_HYDROCHLORIDE', 'ZOLPIDEM', 'LEVOTHYROXINE', 'ENALAPRIL', 'TRIAMTERENE', 'FOSINOPRIL_SODIUM', 'METHYLPHENIDATE_HYDROCHLORIDE', 'AMPHETAMINE__DEXTROAMPHETAMINE', 'SERTRALINE', 'IBUPROFEN', 'ETHINYL_ESTRADIOL__NORETHINDRONE', 'INSULIN_GLARGINE', 'METHYLPHENIDATE', 'NITROGLYCERIN', 'DILTIAZEM_HYDROCHLORIDE', 'PRAVASTATIN', 'LANSOPRAZOLE', 'SALMETEROL_XINAFOATE', 'ROSIGLITAZONE', 'ALENDRONATE_SODIUM', 'RABEPRAZOLE_SODIUM', 'HYDROCHLOROTHIAZIDE__LOSARTAN_POTASSIUM', 'ETHINYL_ESTRADIOL__LEVONORGESTREL', 'ALBUTEROL__IPRATROPIUM', 'ASPIRIN', 'MOMETASONE_NASAL', 'NAPROXEN', 'METOPROLOL_UNSPECIFIED', 'TAMSULOSIN', 'EZETIMIBE__SIMVASTATIN', 'WARFARIN', 'CETIRIZINE_HYDROCHLORIDE', 'CEPHALEXIN', 'ACETAMINOPHEN__PROPOXYPHENE', 'FOSINOPRIL', 'PENICILLIN', 'FEXOFENADINE_HYDROCHLORIDE', 'LORATADINE__PSEUDOEPHEDRINE_SULFATE', 'AMPHETAMINE_ASPARTATE', 'CLOPIDOGREL', 'TRIAMCINOLONE_ACETONIDE', 'AZITHROMYCIN', 'LEVOTHYROXINE_SODIUM', 'OMEPRAZOLE', 'HYDROCHLOROTHIAZIDE__LOSARTAN', 'CAPTOPRIL', 'MECLIZINE_HYDROCHLORIDE', 'NIFEDIPINE', 'TERAZOSIN_HYDROCHLORIDE', 'BUPROPION_HYDROCHLORIDE', 'PROPRANOLOL', 'LATANOPROST', 'ESCITALOPRAM', 'ISOSORBIDE_UNSPECIFIED', 'VENLAFAXINE', 'AMITRIPTYLINE_HYDROCHLORIDE', 'RALOXIFENE_HYDROCHLORIDE', 'INSULIN_ISOPHANE', 'PANTOPRAZOLE', 'TOLTERODINE_TARTRATE', 'ACETAMINOPHEN__OXYCODONE', 'HYDROXYZINE_HYDROCHLORIDE', 'ATORVASTATIN', 'ALENDRONATE', 'AMITRIPTYLINE', 'PHENYTOIN_SODIUM', 'ROSIGLITAZONE_MALEATE', 'RAMIPRIL', 'BECLOMETHASONE_DIPROPIONATE', 'ZOLPIDEM_TARTRATE', 'VALDECOXIB', 'SERTRALINE_HYDROCHLORIDE', 'LOSARTAN', 'PAROXETINE', 'CELECOXIB', 'RABEPRAZOLE', 'TERAZOSIN', 'ACETAMINOPHEN__HYDROCODONE_BITARTRATE', 'ACETAMINOPHEN__CODEINE_PHOSPHATE', 'MONTELUKAST_SODIUM', 'BENAZEPRIL', 'DIVALPROEX_SODIUM', 'AMLODIPINE_BESYLATE', 'LISINOPRIL', 'FLUTICASONE_NASAL', 'CYCLOBENZAPRINE', 'ACETAMINOPHEN__HYDROCODONE', 'LOVASTATIN', 'MOMETASONE_FUROATE_MONOHYDRATE', 'SIMVASTATIN', 'DIGOXIN', 'ACETAMINOPHEN__OXYCODONE_HYDROCHLORIDE', 'INSULIN', 'FELODIPINE', 'ESTROGENS__CONJUGATED__MEDROXYPROGESTERONE_ACETATE', 'CARVEDILOL', 'GEMFIBROZIL', 'ESTROGENS__CONJUGATED', 'CYCLOBENZAPRINE_HYDROCHLORIDE', 'SPIRONOLACTONE', 'METFORMIN_HYDROCHLORIDE', 'VERAPAMIL', 'ATORVASTATIN_CALCIUM', 'FLUVASTATIN_SODIUM', 'CLONIDINE', 'VERAPAMIL_HYDROCHLORIDE', 'FEXOFENADINE', 'RANITIDINE_HYDROCHLORIDE', 'FLUTICASONE__SALMETEROL', 'TRAZODONE_HYDROCHLORIDE', 'PROPRANOLOL_HYDROCHLORIDE', 'CITALOPRAM', 'CETIRIZINE', 'PRAVASTATIN_SODIUM', 'FUROSEMIDE', 'CLOPIDOGREL_BISULFATE', 'EZETIMIBE', 'FLUOXETINE', 'SULFAMETHOXAZOLE__TRIMETHOPRIM', 'AMLODIPINE', 'ALPRAZOLAM', 'DESLORATADINE', 'CITALOPRAM_HYDROBROMIDE')

response_vars[["Physical_Activity_Post_processed"]] <- c('physical_activity')

response_vars[["Phytoestrogens"]] <- c('URXDAZ', 'URXDMA', 'URXGNS', 'URXETL', 'URXEQU', 'URXETD')

response_vars[["Plasma_Fasting_Glucose_and_Insulin"]] <- c('PHAFSTHR', 'PHAFSTMN', 'LBXIN', 'LBXGLU')

response_vars[["Plasma_Glucose"]] <- c('LBXIN', 'LBXCPSI', 'LBXGLU', 'LBXINSI', 'LBXGLUSI')

response_vars[["Polyaromtic_HydrocarbonsPAH"]] <- c('URXP07', 'URXP11', 'URXP19', 'URXP06', 'URXP15', 'URXP17', 'URXP02', 'URXP24', 'URXP10', 'URXP13', 'URXP21', 'URXP04', 'URXP08', 'URXP16', 'URXP01', 'URXP22', 'URXP05', 'URXP14', 'URXP12', 'URXP03', 'URXP20')

response_vars[["Polybrominated_Diphenyl_EthersPBDE"]] <- c('LBXBR3', 'LBXBR5', 'LBXBR8', 'LBXBR6', 'LBXBR2', 'LBXBR7', 'LBXBR66L', 'LBXBR9', 'LBXBB1', 'LBXBR66', 'LBXBR4', 'LBXBR1')

response_vars[["Polyfluorinated_Compounds"]] <- c('LBXMPAH', 'LBXPFHS', 'LBXPFOA', 'LBXPFDO', 'LBXPFSA', 'LBXPFNA', 'LBXPFOS', 'LBXPFUA', 'LBXPFDE', 'LBXEPAH', 'LBXPFHP', 'LBXPFBS')

response_vars[["Polyfluorochemicals_Compounds"]] <- c('LBXMPAH', 'LBXPFHS', 'LBXPFOA', 'LBXPFDO', 'LBXPFSA', 'LBXPFNA', 'LBXPFOS', 'LBXPFUA', 'LBXPFDE', 'LBXEPAH', 'LBXPFHP', 'LBXPFBS')

response_vars[["Priority_Pesticides"]] <- c('URX14D', 'URXOPP', 'URX1TB', 'URXDCB', 'URX3TB')

response_vars[["Prostate_specific_Antigen"]] <- c('LBXP1', 'LBXP2', 'LBDP3')

response_vars[["RBC_Folate_Serum_Folate_and_Vitamin_B12"]] <- c('LBXRBF', 'LBXB12', 'LBXFOL')

response_vars[["Reproductive_related_drugs"]] <- c('how_long_estrogen_progestin', 'age_stopped_birth_control', 'RHQ570', 'how_long_estrogen_progestin_patch', 'taking_birth_control', 'RHQ520', 'how_long_estrogen', 'RHQ572', 'how_long_progestin', 'RHQ596', 'RHQ558', 'RHQ556', 'how_long_estrogen_patch', 'RHQ564', 'RHQ584', 'RHQ510', 'RHQ540', 'RHQ600', 'RHQ574', 'RHQ582', 'RHQ562', 'age_started_birth_control', 'RHQ580', 'RHQ554', 'RHQ598', 'RHQ566')

response_vars[["Serum_Cotinine"]] <- c('LBXCOT')

response_vars[["Smoking_Behavior"]] <- c('SMQ120', 'SMD100FL', 'SMD090', 'SMD057', 'SMD190', 'SMD080', 'SMD650', 'SMQ180', 'SMD030', 'SMQ020', 'SMQ050', 'SMD100CO', 'SMD235', 'SMD075', 'SMQ040', 'SMQ077', 'SMD220', 'SMD055', 'SMQ230', 'SMD100MN', 'SMQ210', 'current_past_smoking', 'SMD160', 'SMD070', 'SMD130', 'SMD100NI', 'SMD641', 'SMQ150', 'SMD100TR', 'cigarette_smoking')

response_vars[["Social_Support"]] <- c('anyone_to_help_social', 'first_degree_support', 'number_close_friends')

response_vars[["Standard_Biochemistry_Profile"]] <- c('LBXSTB', 'LBXSGL', 'LBXSAL', 'LBXSIR', 'LBXSAPSI', 'LBXSNASI', 'LBXSTP', 'LBXSASSI', 'LBXSATSI', 'LBXSGB', 'LBXSKSI', 'LBXSC3SI', 'LBXSGTSI', 'LBXSTR', 'LBXSCRINV', 'LBXSCLSI', 'LBXSUA', 'LBXSCH', 'LBXSCA', 'LBXSCR', 'LBXSPH', 'LBXSOSSI', 'LBXSLDSI', 'LBXSBU')

response_vars[["Street_Drug_Use"]] <- c('DUQ210', 'DUQ130', 'DUQ240', 'DUQ260', 'DUQ370', 'DUQ250', 'DUQ320', 'DUQ100', 'last_time_used_meth', 'DUQ272', 'last_time_used_cocaine', 'DUQ110', 'last_time_used_heroin', 'DUQ200', 'DUQ230', 'DUQ340', 'DUQ352', 'DUQ360', 'DUQ120', 'last_time_used_marijuana', 'DUQ330', 'DUQ280', 'DUQ290', 'DUQ300')

response_vars[["Sun_Exposure"]] <- c('DED038Q')

response_vars[["Supplement_Count"]] <- c('supplement_count')

response_vars[["Supplement_Use"]] <- c('COPPER_Unknown', 'TOTAL_CARBOHYDRATE_gm', 'L_ASPARTIC_ACID_gm', 'ALPHA_TOCOPHEROL_IU', 'CALCIUM_Unknown', 'METHIONINE_mg', 'ALPHA_TOCOPHEROL_mg', 'PHOSPHORUS_mg', 'CAFFEINE_mg', 'COPPER_Trace', 'PROTEIN_gm', 'IRON_Unknown', 'VITAMIN_A_mcg', 'BETA_CAROTENE_Unknown', 'ALANINE_mg', 'GLYCINE_gm', 'OMEGA_6_FATTY_ACIDS_mg', 'MAGNESIUM_Unknown', 'OMEGA_9_FATTY_ACIDS_NA', 'OTHER_FATTY_ACIDS_mg', 'VITAMIN_C_Unknown', 'SODIUM_Trace', 'FOLIC_ACID_mcg', 'MAGNESIUM_PPM', 'TYROSINE_mg', 'SELENIUM_Trace', 'ISOLEUCINE_mg', 'VITAMIN_A_Unknown', 'IRON_mg', 'BETA_CAROTENE_mcg', 'ALPHA_TOCOPHEROL_NA', 'COPPER_mg', 'PHENYLALANINE_mg', 'BETA_CAROTENE__', 'HISTIDINE_mg', 'OMEGA_3_FATTY_ACIDS_Unknown', 'VALINE_mg', 'CHOLESTEROL_mg', 'FOLIC_ACID_Unknown', 'THIAMIN_Unknown', 'CALCIUM_PPM', 'VITAMIN_C_mg', 'SELENIUM_Unknown', 'VITAMIN_A_IU', 'OTHER_FATTY_ACIDS_NA', 'SERINE_mg', 'SELENIUM_mcg', 'DSDCOUNT', 'ARGININE_mg', 'CYSTINE_mg', 'PHOSPHORUS_Unknown', 'THREONINE_mg', 'ARGININE_gm', 'IRON_Trace', 'BETA_CAROTENE_IU', 'DIETARY_FIBER_mg', 'ALCOHOL_PERCENT', 'L_GLUTAMINE_mg', 'GLYCINE_mg', 'VITAMIN_A_mg', 'MAGNESIUM_mg', 'OMEGA_6_FATTY_ACIDS_NA', 'L_ASPARTIC_ACID_mg', 'POTASSIUM_mg', 'RIBOFLAVIN_mg', 'OMEGA_9_FATTY_ACIDS_mg', 'BETA_CAROTENE_mg', 'VITAMIN_B_12_mcg', 'CALCIUM_Trace', 'VITAMIN_B_6_mg', 'TRYPTOPHAN_mg', 'PHOSPHORUS_Trace', 'RIBOFLAVIN_Unknown', 'CALCIUM_mg', 'THIAMIN_mg', 'OTHER_OMEGA_3_FATTY_ACIDS_NA', 'OMEGA_3_FATTY_ACIDS_mg', 'PROLINE_mg', 'VITAMIN_B_12_Unknown', 'LYSINE_mg', 'VITAMIN_B_6_Unknown', 'L_GLUTAMINE_gm', 'LEUCINE_mg', 'SODIUM_mg')

response_vars[["Syphilis"]] <- c('LBDSY3', 'LBXSY1', 'LBDSY4')

response_vars[["Telomere_Mean_and_Standard_Deviation"]] <- c('TELOMEAN')

response_vars[["Thyroid_Stimulating_Hormone_and_Thyroxine"]] <- c('LBXT4', 'LBXTSH')

response_vars[["Total_Arsenics_and_Speciated_Arsenics"]] <- c('URXUDMA', 'URXUTM', 'URXUAB', 'URXUMMA', 'URXUAS5', 'URXUAS3', 'URXUAS', 'URXUAC')

response_vars[["Total_Cholesterol"]] <- c('LBXTC', 'LBDHDL', 'LBDHDD')

#response_vars[["Total_cholesterol"]] <- c('LBXTC', 'LBDHDL')

response_vars[["Total_Mercury_and_Inorganic_Mercury"]] <- c('LBXIHG', 'LBXTHG')

response_vars[["Toxoplasma"]] <- c('LBXTO2', 'LBXTO1')

response_vars[["Transferrin_Receptor"]] <- c('LBXTFR')

response_vars[["Trichomonas___BV"]] <- c('LBXTV', 'LBXBV')

response_vars[["Trichomonas_Vaginalis_Bacterial_Vaginosis"]] <- c('LBXTV', 'LBXBV')

response_vars[["Triglyceride_LDL_Apo_B"]] <- c('LBXAPB', 'LBXTR', 'LBDLDL')

response_vars[["Triglycerides"]] <- c('LBXTR', 'LBDLDL')

response_vars[["Two_Hour_Oral_Glucose_Tolerance_Test"]] <- c('PHAFSTHR', 'LBXGLT', 'PHAFSTMN')

response_vars[["Urinary_Albumin_and_Creatinine"]] <- c('URXUMASI', 'URXUMA', 'URXUCRSI', 'URXUCR', 'URXCRS', 'URXUMS')

#response_vars[["Urinary_albumin_and_creatinine"]] <- c('URXUCRSI', 'URXUMASI', 'URXUCR', 'URXUMA')

response_vars[["Urinary_Arsenics"]] <- c('URXUDMA', 'URXUTM', 'URXUAB', 'URXUMMA', 'URXUAS5', 'URXUAS3', 'URXUAS', 'URXUAC')

response_vars[["Urinary_Chlamydia_and_Gonorrhea"]] <- c('URXUGC', 'URXUCL')

response_vars[["Urinary_Heavy_Metals"]] <- c('URXUBA', 'URXUTU', 'URXUCS', 'URXUBE', 'URXUPT', 'URXUCD', 'URXUMO', 'URXUPB', 'URXUTL', 'URXUCO', 'URXUSB', 'URXUUR')

response_vars[["Urinary_Iodine"]] <- c('URXUIO')

response_vars[["Urinary_Mercury"]] <- c('URXUHG')

response_vars[["Urinary_Organophosphate_Diakyl"]] <- c('URXOP1', 'URXOP5', 'URXOP3', 'URXOP2', 'URXOP4', 'URXOP6')

response_vars[["Urinary_Perchlorate"]] <- c('URXUP8CA', 'URXUP8')

response_vars[["Urinary_Pesticides"]] <- c('URXPCP', 'URXMET', 'URXMMI', 'URXPTU', 'URXSIS', 'URXDCZ', 'URXTCC', 'URXDTZ', 'URXDPY', 'URXMTO', 'URXPPX', 'URXAPE', 'URXOMO', 'URXCBF', 'URXAAZ', 'URXDAM', 'URXETU')

response_vars[["Urinary_Phthalates"]] <- c('URXMOP', 'URXECP', 'URXMOH', 'URXMCP', 'URXMHH', 'URXMNM', 'URXMZP', 'URXMEP', 'URXMBP', 'URXMIB', 'URXMC1', 'URXCNP', 'URXMHP', 'URXCOP', 'URXMNP')

response_vars[["Urinary_Priority_Pesticides"]] <- c('URXHLS', 'URX24D', 'URXFRM', 'URXRIM', 'URXEMM', 'URXMSM', 'URXSSF', 'URXPRO', 'URXSMM', 'URXBSM', 'URXTHF', 'URXTRA', 'URXNOS', 'URXMTM', 'URXTRN', 'URX25T', 'URXPIM', 'URXCHS', 'URXOXS')

response_vars[["Urine_Iodine"]] <- c('URXUIO')

response_vars[["Varicella_Zoster_Virus_Antibody"]] <- c('VARICELL')

response_vars[["Vitamin_A_E_and_Carotenoids"]] <- c('LBXCLZ', 'LBXCBC', 'LBXRPL', 'LBXDTC', 'LBXGTC', 'LBXPHE', 'LBXBCC', 'LBXPHF', 'LBXZEA', 'LBDATCSI', 'LBXRST', 'LBXLCC', 'LBXCLC', 'LBXLUT', 'LBXLYC', 'LBXVIA', 'LBXACY', 'LBXALC', 'LBXBEC', 'LBXVIE', 'LBXCRY', 'LBXLUZ',
              'LBDTLY' #added from below! the only var was unique to below tab
                                                    )

#response_vars[["Vitamin_A_E_and_Carotenoids"]] <- c('LBXCBC', 'LBXBEC', 'LBXRPL', 'LBXGTC', 'LBXLYC', 'LBXVIA', 'LBXVIE', 'LBDTLY', 'LBXCRY', 'LBXLUZ', 'LBXRST', 'LBXALC')

#response_vars[["Vitamins_A_E_and_Carotenoids"]] <- c('LBXCBC', 'LBXBEC', 'LBXRPL', 'LBXGTC', 'LBXLYC', 'LBXVIA', 'LBXVIE', 'LBXCRY', 'LBXLUZ', 'LBXRST', 'LBXALC')

response_vars[["Vitamin_B12"]] <- c('LBXB12')

response_vars[["Vitamin_B6"]] <- c('LBXVB6')

response_vars[["Vitamin_C"]] <- c('LBXVIC')

response_vars[["Vitamin_D_ng_mL"]] <- c('LBXVID')

response_vars[["Volatile_Organic_Compounds"]] <- c('LBXVMC', 'LBXVTC', 'LBXWBM', 'LBXV2T', 'LBXVME', 'LBXVNB', 'LBXZEB', 'LBXV1A', 'LBXV2C', 'LBXVDB', 'LBXV1E', 'LBXZDB', 'LBXZXY', 'LBXV2A', 'LBXZTE', 'LBXV2E', 'LBXVTE', 'LBXVST', 'LBXV4T', 'LBXVEB', 'LBXV4A', 'LBXVDM', 'LBXVBF', 'LBXVCM', 'LBXZCF', 'LBXZTO', 'LBXV4C', 'LBXVDP', 'LBXV3A', 'LBXVCB', 'LBXZBZ', 'LBXZTI', 'LBXV4E', 'LBX2DF', 'LBXV2P', 'LBXWCM', 'LBXVCF', 'LBXVTO', 'LBXZOX', 'LBXWME', 'LBXV3B', 'LBXVBM', 'LBXVXY', 'LBXV1D', 'LBXVBZ', 'LBXZMB', 'LBXVOX', 'LBXWCF', 'LBXVCT', 'LBXWBF', 'LBXVHE')

response_vars[["Volatile_Organic_Compounds_in_Blood_and_Water"]] <- c('LBXVTC', 'LBXWBM', 'LBXVME', 'LBXVDB', 'LBXVST', 'LBXVEB', 'LBXVBF', 'LBXVCM', 'LBXV4C', 'LBXV3A', 'LBXWCM', 'LBXVTO', 'LBXVCF', 'LBXWME', 'LBXVBM', 'LBXVXY', 'LBXVBZ', 'LBXVOX', 'LBXWCF', 'LBXVCT', 'LBXWBF')

response_vars[["Volatile_Organic_Compounds_in_Water_and_Related_Questionnaire_Items"]] <- c('LBXWBF', 'LBXWME', 'LBXWBM', 'LBXWCF', 'LBXWCM')

response_vars[["sexual_transmitted_disease"]] <- c('SXQ265', 'SXQ280', 'SXQ020', 'SXQ270', 'SXQ260')

response_vars[["tuberculosis"]] <- c('TBQ040', 'TBQ060', 'TBQ020', 'TBQ030', 'TBQ050')