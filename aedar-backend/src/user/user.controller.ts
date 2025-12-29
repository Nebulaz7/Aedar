import { Controller, Get, UseGuards, Req } from '@nestjs/common';
import { SupabaseAuthGuard } from '../common/guard/auth.guard';
import type { Request } from 'express';

@Controller('user')
export class UserController {
  @UseGuards(SupabaseAuthGuard)
  @Get('me')
  getMe(@Req() req: Request) {
    return {
      message: 'Authenticated user',
      user: req['user'],
    };
  }
}