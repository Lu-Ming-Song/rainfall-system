from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, Role, Permission, SystemSetting

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'phone', 'role', 'is_staff', 'is_active')
    list_filter = ('is_staff', 'is_active', 'role')
    fieldsets = UserAdmin.fieldsets + (
        ('扩展信息', {'fields': ('phone', 'role')}),
    )

@admin.register(Role)
class RoleAdmin(admin.ModelAdmin):
    list_display = ('name', 'desc', 'created_time', 'updated_time')
    filter_horizontal = ('permissions',)
    search_fields = ('name', 'desc')

@admin.register(Permission)
class PermissionAdmin(admin.ModelAdmin):
    list_display = ('name', 'code', 'desc', 'created_time', 'updated_time')
    search_fields = ('name', 'code', 'desc')

@admin.register(SystemSetting)
class SystemSettingAdmin(admin.ModelAdmin):
    list_display = ('key', 'desc', 'created_time', 'updated_time')
    search_fields = ('key', 'desc')
