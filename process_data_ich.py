import csv


def read_csv_into_list(file_path):

    entry_list = []

    with open(file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row_num, row in enumerate(spamreader):
            if row_num is not 0:
                entry_list.append(row)
                # print(', '.join(row))

    return entry_list

def get_cases():

    file_path = "/collab/gianca-group/icoronado/ct_stroke/MachineLearningInAcu-HemorrhagicStroke_DATA_2020-05-19_2004.csv"

    entry_list = read_csv_into_list(file_path)

    # print(entry_list)
    patient_dict = {}

    for file_p in entry_list:
        part_file = file_p[0].split(",")
        patient_id, stroke_check = part_file[0], part_file[5]
        check_non_ich = False

        for entry in range(7, len(part_file)-1):
            list_non_ich = part_file[8:12]
            # print(list_non_ich)
            try:
                list_non_ich = [int(x) for x in list_non_ich]
                check_non_ich = any(list_non_ich)
                

            except ValueError:
                continue

        #     print(part_file[entry], end=" ", flush=False)
        # print()

        if stroke_check == '':
            stroke_check = 0
            patient_dict[patient_id.lower()] = int(stroke_check)
            # print("Case ID: {0} | Non_ICH_List: {1} | Non_ICH_Check: {2}".format(patient_id, list_non_ich, check_non_ich))

        elif int(stroke_check) > 0 and check_non_ich == False:
            stroke_check = 1
            patient_dict[patient_id.lower()] = int(stroke_check)
            # print("Case ID: {0} | Non_ICH_List: {1} | Non_ICH_Check: {2}".format(patient_id, list_non_ich, check_non_ich))
        
        

    # print(patient_dict)

    return patient_dict

def main():
    get_cases()

if __name__ == '__main__':
    main()