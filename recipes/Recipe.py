from dataclasses import dataclass, fields

@dataclass
class Recipe:
    def __post_init__(self):
        for field in fields(self):
            field_recipe_father = getattr(self.__class__.__base__, field.name, None)
            field_recipe_child = getattr(self.__class__, field.name, None)
            field_init = getattr(self, field.name, None)
            
            if field_init is not None and field_init != field_recipe_father:
                setattr(self, field.name, field_init)
            elif field_recipe_child is not None and field_recipe_child != field_recipe_father:
                setattr(self, field.name, field_recipe_child)
            else:
                setattr(self, field.name, field_recipe_father)
