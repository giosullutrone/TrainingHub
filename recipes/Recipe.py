from dataclasses import dataclass, fields

@dataclass
class Recipe:
    def __post_init__(self):
        for field in fields(self):
            field_recipe_father = getattr(self.__class__.__base__, field.name, None)
            field_recipe_child = getattr(self.__class__, field.name)
            field_init = getattr(self, field.name)
            
            if field_init != field_recipe_father:
                setattr(self, field.name, field_init)
            elif field_recipe_child != field_recipe_father:
                setattr(self, field.name, field_recipe_child)