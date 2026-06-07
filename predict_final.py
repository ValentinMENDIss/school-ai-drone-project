from ultralytics import YOLO
from globals import *

class KI:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        ''' retrieve the model's classes names'''
        self.class_names = self.model.names

        #self.img = "downward_foto.jpg"

        self.tile_data_list = []
        self.tile_cls_data_list = []

        self.object_data_list = []
        self.object_cls_data_list = []

        self.cls_tile_ids = [2, 3, 5, 6, 7, 8]


    def model_predict(self):
        results = self.model.predict(source=f"{IMG_DIR}/{IMG}", show=True, save=True)

        for result in results:
            ''' get position + class ids; Translate numpy's arrays into regular lists '''
            obb_pos = result.obb.xyxy.tolist()
            obb_cls = result.obb.cls.tolist()

            self.get_obb_data_list(obb_cls, obb_pos)

            self.tile_data_list = self.sort_obb_data(self.tile_data_list)
            self.object_data_list = self.sort_obb_data(self.object_data_list)

            result.save(filename="export/labeled-images/labeled_image.jpg")


    def get_obb_data_list(self, cls, pos):

        def check_obb_is_tile(cls):
            for tile in self.cls_tile_ids:
                if cls == tile:
                    return True

        def tile_obb_data_add_to_list(id, cls, pos):
            self.tile_data_list.append([])
            self.tile_data_list[id].append(cls)
            self.tile_data_list[id].append(pos)

            # add to cls_list
            self.tile_cls_data_list.append(cls)

            id += 1
            return id

        def object_obb_data_add_to_list(id, cls, pos):
            self.object_data_list.append([])
            self.object_data_list[id].append(cls)
            self.object_data_list[id].append(pos)

            # add to cls_list
            self.object_cls_data_list.append(cls)

            id += 1
            return id


        tile_id = 0
        object_id = 0
        for cls, pos in zip(cls, pos):
            if check_obb_is_tile(cls):
                tile_id = tile_obb_data_add_to_list(tile_id, cls, pos)
            else:
                object_id = object_obb_data_add_to_list(object_id, cls, pos)
            

    def sort_obb_data(self, data_list):
        
        def sort_obb_data_y_pos():
            buffer_list = sorted(data_list, key=lambda box: box[1][1])
            return buffer_list

        def sort_obb_data_in_rows():
            
            def create_new_row(ls, id_row):
                ls.append([])
                id_row += 1
                id_in_row = 0
                return ls, id_row, id_in_row


            row_sensitivity_difference_px = 20
            # nested list
            buffer_list = [[]]

            last_obb_y_coord = float('inf')
            id_row = 0
            id_in_row = 0
            for box in data_list:
                current_obb_y_coord = box[1][1]
                obb_cls = box[0]
                obb_pos = box[1]

                if last_obb_y_coord < current_obb_y_coord - row_sensitivity_difference_px:
                    buffer_list, id_row, id_in_row = create_new_row(buffer_list, id_row)

                if last_obb_y_coord == float('inf'):
                    last_obb_y_coord = current_obb_y_coord


                buffer_list[id_row].append([])
                buffer_list[id_row][id_in_row].append(obb_cls)
                buffer_list[id_row][id_in_row].append(obb_pos)
                id_in_row += 1

                last_obb_y_coord = current_obb_y_coord

            return buffer_list 

        def sort_obb_data_x_pos():
            #list comprehension - based on the values of previous list create new one (with shorter syntax)
            buffer_list = [sorted(row, key=lambda box: box[1][0]) for row in data_list]
            return buffer_list

        def convert_obb_data_list_to_class_matrix():
            buffer_matrix = []

            id = 0
            for row in data_list:
                buffer_matrix.append([])
                for item in row:
                    cls = item[0]
                    buffer_matrix[id].append(self.class_names[cls])
                id +=1

            return buffer_matrix


        data_list = sort_obb_data_y_pos()
        data_list = sort_obb_data_in_rows()
        data_list = sort_obb_data_x_pos()
        data_list = convert_obb_data_list_to_class_matrix()

        return data_list


    def return_tiles_type_percent(self):

        def generate_tiles_type_count_matrix():
            matrix = [[]]

            previous_cls = None
            current_cls_count = 0

            id_row = 0
            for cls in sorted(self.tile_cls_data_list):
                current_cls_count += 1

                if previous_cls == None:
                    previous_cls = cls

                if previous_cls != cls:
                    id_row += 1
                    matrix.append([])
                    current_cls_count = 1
                    previous_cls = cls


                matrix[id_row] = [self.class_names[cls], current_cls_count]

            return matrix

        def convert_tiles_count_matrix_to_percent(types_count_matrix):
            sum_tiles = 0
            for tile in types_count_matrix:
                sum_tiles += tile[1]

            for tile in types_count_matrix:
                # get percents instead of count; + round to 2 decimal points (x.yz)
                tile[1] = round((tile[1] / sum_tiles * 100), 2)

            return types_count_matrix


        count_matrix = generate_tiles_type_count_matrix()
        percent_matrix = convert_tiles_count_matrix_to_percent(count_matrix)

        return percent_matrix


    def return_object_type_count(self):

        def generate_object_count_matrix():
            object_percent_list = [[]]

            previous_cls = None
            current_cls_count = 0

            id_row = 0
            for cls in sorted(self.object_cls_data_list):
                current_cls_count += 1

                if previous_cls == None:
                    previous_cls = cls

                if previous_cls != cls:
                    id_row += 1
                    object_percent_list.append([])
                    current_cls_count = 1
                    previous_cls = cls


                object_percent_list[id_row] = [self.class_names[cls], current_cls_count]

            return object_percent_list


        types_count_matrix = generate_object_count_matrix()

        return types_count_matrix

    def get_tile_map(self):
        return self.tile_data_list


def predict_test():
    ki = KI(model_path="best2.pt")

    ki.model_predict()

    print(f"MAP: {ki.get_tile_map()}")
    print(f"TILES TYPE PERCENT: {ki.return_tiles_type_percent()}")
    print(f"OBJECT TYPE COUNT: {ki.return_object_type_count()}")


#predict_test()